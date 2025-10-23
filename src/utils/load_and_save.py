import os
import hashlib
from safetensors.torch import load_file, save_file
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_cache_path(image_path: str, cache_dir: str = './image_info_cache', concept_name: str = None) -> str:
    with open(image_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    
    file_path = f"{concept_name}_{sha}.safetensors" if concept_name else f"{sha}.safetensors"
    return os.path.join(cache_dir, file_path)

def load_image_info(image_path: str, cache_dir: str = './image_info_cache', concept_name: str = None):
    try:
        cache_path = get_cache_path(image_path, cache_dir, concept_name)
        if not Path(cache_path).is_file():
            return None
        
        return load_file(cache_path, device='cpu')
    except Exception:
        return None

def save_image_info(image_info, image_path: str, cache_dir: str = './image_info_cache', concept_name: str = None):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    cpu_info = {}
    for key, value in image_info.items():
        if isinstance(value, torch.Tensor):
            cpu_info[key] = value.cpu()
        else:
            cpu_info[key] = value
    
    save_file(cpu_info, get_cache_path(image_path, cache_dir, concept_name))

def save_image(pil_image, cache_dir='./image_cache', suffix=""):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    image_bytes = pil_image.tobytes()
    sha = hashlib.sha256(image_bytes).hexdigest()
    
    filename = f'{sha}_{suffix}.png' if suffix else f'{sha}.png'
    image_path = os.path.join(cache_dir, filename)
    
    pil_image.save(image_path, optimize=True)
    return image_path

def load_image(image_path=None, pil_image=None, to_tensor=False):
    if image_path is not None:
        image_pil = Image.open(image_path).convert("RGB")
    elif pil_image is not None:
        image_pil = pil_image
    else:
        raise ValueError("Either image_path or pil_image must be provided")
    
    if to_tensor:
        image_tensor = _DEFAULT_TRANSFORM(image_pil)
        return image_pil, image_tensor  
    else:
        return image_pil