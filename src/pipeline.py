import os
import shutil
from typing import List
from dataclasses import dataclass, field
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from einops import rearrange
import gc

from flux.sampling import denoise_fireflow as denoise_fn
from flux.sampling import prepare, get_schedule, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5

from core.collage_creation import CollageCreator
from utils.load_and_save import load_image, save_image, load_image_info, save_image_info

@dataclass
class ConceptConfig:
    class_name: str
    image_path: str
    scale: float = field(default=1.)
    x_bias: int = field(default=0)
    y_bias: int = field(default=0)
    angle: float = field(default=0.)
    flip: bool = field(default=False)
    alignment: str = field(default="center")

@dataclass
class GenerationConfig:
    seed: int = 0
    num_steps: int = 25
    guidance: float = 3.0
    width: int = 1024
    height: int = 1024
    start_inject_step: int = 0
    end_inject_step: int = 25
    inject_block_ids: List[int] = field(default_factory=lambda: list(range(0, 57)))
    sim_threshold: float = 0.2
    cyc_threshold: float = 1.5
    inject_match_dropout: float = 0.2

class FreeGraftorPipeline:
    def __init__(self, models=None, device="cuda", image_cache_dir='./image_cache', image_info_cache_dir='./image_info_cache', requires_offload=False):
        self.device = device
        self.requires_offload = requires_offload
        
        # self.tracker = MemTracker()
        
        
        if self.requires_offload:
            device = 'cpu'
        else:
            device = self.device
        
        if models is None:
            models = {}
            
        self.load_flux(models, device=device)
        
        self.image_cache_dir = image_cache_dir
        self.image_info_cache_dir = image_info_cache_dir
        
        self.collage_creator = CollageCreator(models=models, device=self.device, 
                                              image_cache_dir=image_cache_dir, image_info_cache_dir=image_info_cache_dir, 
                                              requires_offload=requires_offload)
        

    def load_flux(self, models, device='cpu'):
        model_loaders = {
            't5': lambda: load_t5(os.getenv('FLUX_DEV'), device, max_length=512),
            'clip': lambda: load_clip(os.getenv('FLUX_DEV'), device),
            'flow': lambda: load_flow_model('flux-dev', self.device),
            'ae': lambda: load_ae('flux-dev', device)
        }
     
        for name, loader in model_loaders.items():
            if name in models:
                setattr(self, name, models[name])
            else:
                model = loader()
                models[name] = model
                setattr(self, name, model)
            print(f"{name} loaded to {device}")
            
    def onload(self, modules=['flow', 't5', 'clip', 'ae', 'collage_creator']):
        print('onload', modules)
        for module in modules:
            if module == 'collage_creator':
                self.collage_creator.onload()
            elif hasattr(self, module):
                setattr(self, module, getattr(self, module).to(self.device))
        
        
    def offload(self, modules=['flow', 't5', 'clip', 'ae', 'collage_creator']):
        print('offload', modules)
        for module in modules:
            if module == 'collage_creator':
                self.collage_creator.offload()
            elif hasattr(self, module):
                setattr(self, module, getattr(self, module).to('cpu'))
        
        torch.cuda.empty_cache()
        
    @torch.inference_mode()
    def generate_template(self, prompt: str, config: GenerationConfig, callback=None):
        torch.manual_seed(config.seed)
        dtype = torch.bfloat16
        init_noise = torch.randn((1, config.height * config.width // 256, 64), 
                                device=self.device, dtype=dtype)
        x = torch.randn((1, 16, config.height // 8, config.width // 8), 
                       device=self.device, dtype=dtype)
        
        inp_gen = prepare(self.t5, self.clip, x, prompt=prompt)
        
        inp_gen['img'] = init_noise
        timesteps = get_schedule(config.num_steps, inp_gen["img"].shape[1], shift=True)
        
        gen_x = denoise_fn(self.flow, **inp_gen, timesteps=timesteps, guidance=config.guidance, 
                          inverse=False, config=config, latent_storage={}, callback=callback)
            
        gen_x = unpack(gen_x.float(), config.height, config.width)
        
        if self.requires_offload:
            self.onload(['ae'])
        
        with torch.autocast(device_type=self.device, dtype=dtype):
            x = self.ae.decode(gen_x)  
        if self.requires_offload:
            self.offload(['ae'])
        x = x.clamp(-1, 1).float()
        x = rearrange(x[0], "c h w -> h w c") 
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, inp_gen
    
    def create_collage(self, prompt, concept_configs, config: GenerationConfig, template_path=None, callback=None):
        if template_path and Path(template_path).is_file():
            template = Image.open(template_path).convert('RGB')
            inp_gen = None
        else:
            
            template, inp_gen = self.generate_template(prompt, config, callback=callback)
            template_path = save_image(template, self.image_cache_dir, suffix='template')
        
        collage, collage_mask = self.collage_creator(template, concept_configs)
        collage_mask_array = np.array(collage_mask, dtype=np.uint8) // 255
        collage_mask_tensor = torch.from_numpy(collage_mask_array).unsqueeze(0).unsqueeze(0)
        collage_mask_tensor = collage_mask_tensor.to(device=self.device, dtype=torch.bfloat16)
        
        target_size = (collage_mask_tensor.shape[2]//16, collage_mask_tensor.shape[3]//16)
        collage_mask_tensor = torch.nn.functional.interpolate(
            collage_mask_tensor.float(), size=target_size, mode='nearest'
        ).to(torch.bfloat16).flatten()          
        
        
        return collage, collage_mask_tensor, inp_gen
    
    @torch.inference_mode()
    def invert(self, image_path=None, pil_image=None, prompt="", config: GenerationConfig=None, 
               latent_storage=None, callback=None):
        pil_image = load_image(image_path, pil_image)
        x = self.encode_image(pil_image)
        
        inp_inv = prepare(self.t5, self.clip, x, prompt=prompt)
        
        timesteps = get_schedule(config.num_steps, inp_inv["img"].shape[1], shift=True)
        
        image_info = {}
        z = denoise_fn(self.flow, **inp_inv, timesteps=timesteps, guidance=1, inverse=True, 
                      config=config, latent_storage=image_info, callback=callback)
            
        image_info.update({
            'z': z.cpu(),
            'x': x.cpu(), 
            'image': torch.tensor(np.array(pil_image)),
            'txt': inp_inv['txt'].cpu(),
            'vec': inp_inv['vec'].cpu(),
        })

        return image_info
            
    @torch.inference_mode()
    def encode_image(self, image):
        image_array = np.array(image)
        image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 127.5 - 1
        image = image.unsqueeze(0).to(self.device)
        if self.requires_offload:
            self.onload(['ae'])
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            image = self.ae.encode(image).to(torch.bfloat16)
        if self.requires_offload:
            self.offload(['ae'])
        return image    
    
    @torch.inference_mode()
    def invert_and_record(self, image_path=None, pil_image=None, config: GenerationConfig=None, callback=None):
        if image_path is None:
            image_path = save_image(pil_image, self.image_cache_dir)
        image_info = load_image_info(image_path, self.image_info_cache_dir)
        if not (image_info and 'z' in image_info):
            image_info = self.invert(image_path=image_path, prompt="", config=config, callback=callback)
            save_image_info(image_info, image_path, self.image_info_cache_dir)
        return image_info
    
    @torch.inference_mode()
    def final_generation(self, prompt, all_image_info, config: GenerationConfig, inp_gen=None, callback=None):
        init_noise = all_image_info[0]['z'].to(self.device)
        x = torch.randn((1, 16, config.height // 8, config.width // 8), 
                       device=self.device, dtype=torch.bfloat16)
        
        if inp_gen is None:
            inp_gen = prepare(self.t5, self.clip, x, prompt=prompt)
        inp_gen['ref_vecs'] = [item['vec'].to(self.device) for item in all_image_info]
        inp_gen['ref_txts'] = [item['txt'].to(self.device) for item in all_image_info]
        inp_gen['img'] = init_noise
        inp_gen['ref_imgs'] = [item['z'].to(self.device) for item in all_image_info]
        inp_gen['ref_masks'] = [item['mask'].to(self.device) for item in all_image_info]
        
        timesteps = get_schedule(config.num_steps, inp_gen["img"].shape[1], shift=True)
        
        gen_x, ref_x_recons = denoise_fn(self.flow, **inp_gen, timesteps=timesteps, guidance=config.guidance, inverse=False, config=config, latent_storage=all_image_info[0], callback=callback)
        
        gen_x = unpack(gen_x.float(), config.width, config.height)
        if self.requires_offload:
            self.onload(['ae'])
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            x = self.ae.decode(gen_x)
        if self.requires_offload:
            self.offload(['ae'])
        x = x.clamp(-1, 1).float()
        x = rearrange(x[0], "c h w -> h w c")
        gen_image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            
        return gen_image
    
    def __call__(
        self,
        concept_configs: List[ConceptConfig],
        prompt: str,
        template_prompt: str = None,
        template_path: str = None,
        output_dir: str = 'inference_results', 
        clear_image_cache: bool = False,
        clear_image_info_cache: bool = False,
        config: GenerationConfig = None,
        callback=None
    ):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        if config is None:
            config = GenerationConfig()
        
        
        
        try:
            if not template_prompt:
                template_prompt = prompt
            
            collage, collage_mask_tensor, inp_gen = self.create_collage(template_prompt, concept_configs, config, template_path, callback=callback)
            collage_path = save_image(collage, self.image_cache_dir, "collage")
            
            collage_info = self.invert_and_record(image_path=collage_path, config=config, callback=lambda x,y: callback(x+config.num_steps, y) if callback else None)
            
            collage_info['mask'] = collage_mask_tensor.cpu()
            
            if prompt != template_prompt:
                inp_gen = None
            gen_image = self.final_generation(prompt, [collage_info], config,  inp_gen=inp_gen, callback=lambda x,y: callback(x+config.num_steps*2, y) if callback else None)
            
            seed = config.seed
            save_image(gen_image, output_dir, f"seed{seed}")
            
            del collage_info
            
        finally:
            if clear_image_cache and os.path.exists(self.image_cache_dir):
                shutil.rmtree(self.image_cache_dir, ignore_errors=True)
            if clear_image_info_cache and os.path.exists(self.image_info_cache_dir):
                shutil.rmtree(self.image_info_cache_dir, ignore_errors=True)
            
            gc.collect()
            torch.cuda.empty_cache()
            
            
        
        
        return gen_image