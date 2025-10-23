import sys
sys.path.append('../')

import os
from collections import defaultdict, deque
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms

from segment_anything import sam_model_registry, SamPredictor
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.load_and_save import save_image, load_image_info, save_image_info, load_image


class CollageCreator:
    def __init__(self, models=None, device='cuda:0', image_cache_dir='./image_cache', image_info_cache_dir='./image_info_cache', requires_offload=False):
        self.device = device
        self.image_cache_dir = image_cache_dir
        self.image_info_cache_dir = image_info_cache_dir
        self.requires_offload = requires_offload
        if self.requires_offload:
            device = 'cpu'
        else:
            device = self.device

        self._load_models(models, device=self.device)
    
    def _load_models(self, models, device='cpu'):
        if models is None:
            models = {}
            
        if 'lama' not in models:
            models['lama'] = pipeline(Tasks.image_inpainting, model='iic/cv_fft_inpainting_lama', 
                                    refine=True, device=self.device, dtype=torch.bfloat16)
        self.lama = models['lama']
        
        if 'grounding_dino_processor' not in models:
            models['grounding_dino_processor'] = AutoProcessor.from_pretrained(os.getenv('GROUNDING_DINO'))
        self.grounding_dino_processor = models['grounding_dino_processor']
        
        if 'grounding_dino' not in models:
            models['grounding_dino'] = AutoModelForZeroShotObjectDetection.from_pretrained(
                os.getenv('GROUNDING_DINO')).to(device=device, dtype=torch.bfloat16)
        self.grounding_dino = models['grounding_dino']
        
        if 'sam' not in models:
            sam_predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=os.getenv('SAM')))
            sam_predictor.model = sam_predictor.model.to(device=device, dtype=torch.bfloat16)
            models['sam'] = sam_predictor
        self.sam = models['sam']
        
        self._optimize_models()
    
    def _optimize_models(self):
        if hasattr(self.grounding_dino, 'eval'):
            self.grounding_dino.eval()
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def offload(self):
        self.grounding_dino = self.grounding_dino.to('cpu')
        self.sam.model = self.sam.model.to('cpu')
        self.lama.infer_model = self.lama.infer_model.to('cpu')
        torch.cuda.empty_cache()
    
    def onload(self):
        self.grounding_dino = self.grounding_dino.to(self.device)
        self.sam.model = self.sam.model.to(self.device)
        self.lama.infer_model = self.lama.infer_model.to(self.device)
    
    @torch.inference_mode()
    def grounding(self, image_path=None, pil_image=None, prompt="", box_threshold=0.35, text_threshold=0.25):
        image = load_image(image_path, pil_image)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
            inputs = self.grounding_dino_processor(images=image, text=[[prompt]], return_tensors="pt").to(self.device)
            outputs = self.grounding_dino(**inputs)
            
        results = self.grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)]
        )
        result = results[0]
        return result
    
    @torch.inference_mode()
    def segment(self, image_path=None, pil_image=None, prompt=None):
        image_pil = load_image(image_path, pil_image)
        
        outputs = self.grounding(pil_image=image_pil, prompt=prompt)
        boxes = outputs['boxes'].cpu()
        scores = outputs['scores'].cpu()
        
        if image_path is not None: 
            cv2_image = cv2.imread(image_path)
        else:
            cv2_image = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
        
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        self.sam.set_image(image)
        
        if len(boxes) > 0:
            transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image.shape[:2]).to(self.device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
                masks, _, _ = self.sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )
            
            combined = list(zip(boxes, masks, scores))
            combined.sort(key=lambda x: x[2], reverse=True)
            
            mask_list, box_list = zip(*[(item[1], item[0]) for item in combined])
            return list(mask_list), list(box_list)
        else:
            return [], []
    
    def erase(self, image_path=None, mask_path=None, pil_image=None, pil_mask=None):
        if image_path is None:
            image_path = save_image(pil_image, self.image_cache_dir, 'erasion_input')
        if mask_path is None:
            mask_path = save_image(pil_mask, self.image_cache_dir, 'erasion_mask')
        
        inputs = {
            'img': image_path,
            'mask': mask_path
        }
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.startswith('cuda')):
        #     outputs = self.lama(inputs)
        outputs = self.lama(inputs)
        erased_array_bgr = outputs[OutputKeys.OUTPUT_IMG]
        erased_array_rgb = erased_array_bgr[..., ::-1]
        erased_image = Image.fromarray(erased_array_rgb).resize(pil_image.size)
        return erased_image
    
    def rotate_img(self, img, angle):
        h, w = img.shape[:2]
        rotate_center = (w/2, h/2)
        
        # Get rotation matrix with center, angle and scale
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        
        # Calculate new boundaries
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        
        # Adjust rotation matrix for translation
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img
    
    def paste_image(self, img1, mask1, img2, mask2, scale=1.0, x_bias=0, y_bias=0, angle=0, flip=False, alignment="center"):
        # Convert PIL.Image to numpy arrays
        img1 = np.array(img1)
        img2 = np.array(img2)
        mask1 = np.array(mask1) // 255
        mask2 = np.array(mask2) // 255
        
        if flip:
            img2 = cv2.flip(img2, 1)
            mask2 = cv2.flip(mask2, 1)
        
        if angle > 0:
            img2 = self.rotate_img(img2, angle)
            mask2 = self.rotate_img(mask2, angle)

        # Calculate area and centroid of object 1
        area1 = np.sum(mask1)
        M = cv2.moments(mask1)
        cx1 = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy1 = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

        # Calculate area of object 2
        area2 = np.sum(mask2)

        # Calculate scale factor
        scale_factor = np.sqrt(area1 / area2) * scale

        # Scale object 2 proportionally
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        new_h2, new_w2 = int(h2 * scale_factor), int(w2 * scale_factor)
        resized_img2 = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_LINEAR)
        resized_mask2 = cv2.resize(mask2, (new_w2, new_h2), interpolation=cv2.INTER_NEAREST)

        # Calculate centroid of scaled object 2
        M_resized = cv2.moments(resized_mask2)
        cx2 = int(M_resized['m10'] / M_resized['m00']) if M_resized['m00'] != 0 else 0
        cy2 = int(M_resized['m01'] / M_resized['m00']) if M_resized['m00'] != 0 else 0

        # Calculate paste position
        # Get boundary coordinates of object 1
        ys1, xs1 = np.where(mask1)
        if len(ys1) > 0:
            top1, bottom1 = ys1.min(), ys1.max()
            left1, right1 = xs1.min(), xs1.max()
        else:
            top1 = bottom1 = left1 = right1 = 0

        # Get boundary coordinates of scaled object 2
        ys2, xs2 = np.where(resized_mask2)
        if len(ys2) > 0:
            top2, bottom2 = ys2.min(), ys2.max()
            left2, right2 = xs2.min(), xs2.max()
        else:
            top2 = bottom2 = left2 = right2 = 0

        # Adjust offset based on alignment
        x_offset = cx1 - cx2 + x_bias
        y_offset = cy1 - cy2 + y_bias

        # Handle vertical alignment
        if 'top' in alignment:
            y_offset = top1 - top2 + y_bias
        elif 'bottom' in alignment:
            y_offset = bottom1 - bottom2 + y_bias

        # Handle horizontal alignment
        if 'left' in alignment:
            x_offset = left1 - left2 + x_bias
        elif 'right' in alignment:
            x_offset = right1 - right2 + x_bias

        # Create img3 and mask3
        img3 = img1.copy()
        mask3 = np.zeros_like(mask1)

        # Paste scaled object 2 into img1
        y1, y2 = max(y_offset, 0), min(y_offset + new_h2, img1.shape[0])
        x1, x2 = max(x_offset, 0), min(x_offset + new_w2, img1.shape[1])

        y1_resized = max(-y_offset, 0)
        x1_resized = max(-x_offset, 0)
        y2_resized = y1_resized + (y2 - y1)
        x2_resized = x1_resized + (x2 - x1)

        resized_img2_cropped = resized_img2[y1_resized:y2_resized, x1_resized:x2_resized]
        resized_mask2_cropped = resized_mask2[y1_resized:y2_resized, x1_resized:x2_resized]
        resized_mask2_cropped_ = resized_mask2_cropped.copy()
        
        resized_mask2_cropped_ = np.expand_dims(resized_mask2_cropped_, -1)

        img3[y1:y2, x1:x2] = resized_img2_cropped * resized_mask2_cropped_ + img3[y1:y2, x1:x2] * (1 - resized_mask2_cropped_)
        mask3[y1:y2, x1:x2] = resized_mask2_cropped

        # Convert result back to PIL.Image
        img3 = Image.fromarray(img3)
        mask3 = Image.fromarray(mask3 * 255)

        return img3, mask3
    
    def __call__(self, template, concept_configs, mask_iou_threshold=0.8):
        erasion_mask_array = np.zeros((template.size[1], template.size[0]), dtype=np.uint8)
        instance_count = defaultdict(int)
        all_image_info = []
        
        for config in concept_configs:
            instance_count[config.class_name] += 1
            image_info = load_image_info(config.image_path, self.image_info_cache_dir, config.class_name)
            if image_info:
                all_image_info.append(image_info)
            else:
                image_info = {}
                mask_list, bbox_list = self.segment(image_path=config.image_path, prompt=config.class_name)
                mask = mask_list[0]
                bbox = torch.tensor([int(m) for m in bbox_list[0]], dtype=torch.int32)
                mask_raw = mask.squeeze(0).cpu()
                image_info['mask_raw'] = mask_raw
                mask = mask.unsqueeze(dim=0).to(dtype=torch.bfloat16, device='cuda')
                mask = transforms.Resize((mask.shape[2]//16, mask.shape[3]//16))(mask)
                mask = mask.flatten()  
                image_info['mask'] = mask.cpu()
                image_info['bbox'] = bbox.cpu()
                save_image_info(image_info, config.image_path, self.image_info_cache_dir, config.class_name)
                all_image_info.append(image_info)
            
        detections = {}
        detected_masks = []
        for class_name, n_objs in instance_count.items():
            if not class_name in detections:
                detections[class_name] = deque()
            mask_list, bbox_list = self.segment(pil_image=template, prompt=class_name)
            # for idx in range(n_objs):
            idx = 0
            count = 0
            while count < n_objs and idx < len(mask_list):
                mask, bbox = mask_list[idx], bbox_list[idx]
                tar_bbox = torch.tensor([int(m) for m in bbox], dtype=torch.int32)
                tar_mask_array = mask.squeeze(dim=0).cpu().numpy().astype('uint8')
                overlap = False
                if len(detected_masks):
                    for detected_mask in detected_masks:
                        intersection = (tar_mask_array * detected_mask).sum()
                        union = tar_mask_array + detected_mask
                        union[union>1] = 1
                        union = union.sum()
                        iou = intersection / union
                        if iou > mask_iou_threshold:
                            overlap = True
                            break
                    if overlap:
                        idx += 1
                        continue
                detected_masks.append(tar_mask_array)
                tar_mask = Image.fromarray(tar_mask_array*255).convert('L')
                erasion_mask_array += tar_mask_array
                detections[class_name].append((tar_mask, tar_bbox))
                idx += 1
                count += 1
                
        erasion_mask_array[erasion_mask_array > 0] = 1        
        area = np.sum(erasion_mask_array)
        ksize = int(np.sqrt(area) * 0.2 + 0.5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        erasion_mask_array = cv2.dilate(erasion_mask_array, kernel, iterations=1)
        erasion_mask = Image.fromarray(erasion_mask_array*255).convert('L')
        
        template_erased = self.erase(pil_image=template, pil_mask=erasion_mask)
        save_image(template_erased, self.image_cache_dir, suffix='erased')
        
        template_mask = Image.new('L', template_erased.size, color='black')
        template_pasted = template_erased
        for config, image_info in zip(concept_configs, all_image_info):
            if not len(detections[config.class_name]):
                break
            tar_mask, tar_bbox = detections[config.class_name].popleft()
            ref_image = Image.open(config.image_path).convert('RGB')
            ref_mask = Image.fromarray(image_info['mask_raw'].numpy().astype('uint8')*255).convert('L')
            template_pasted, new_mask = self.paste_image(template_pasted, tar_mask, ref_image, ref_mask, 
                                                         scale=config.scale, x_bias=config.x_bias, y_bias=config.y_bias,
                                                         angle=config.angle, flip=config.flip, alignment=config.alignment)
            template_mask_array = np.array(template_mask)
            new_mask_array = np.array(new_mask)
            template_mask_array = np.maximum(template_mask_array, new_mask_array)
            template_mask = Image.fromarray(template_mask_array)
        
        return template_pasted, template_mask