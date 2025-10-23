import math
from typing import Callable, List

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

from tqdm import tqdm


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        
        img = img + (t_prev - t_curr) * pred

    return img, info


def denoise_rf_solver(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def denoise_fireflow(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    config=None,  # GenerationConfig instance
    latent_storage: dict = None,  # for storing latents
    guidance: float = 4.0,
    ref_imgs: List[Tensor] = None, 
    ref_txts: List[Tensor] = None,
    ref_vecs: List[Tensor] = None,
    ref_masks: List[Tensor] = None,
    store_latents: bool = True,
    callback=None
):
    if latent_storage is None:
        latent_storage = {}
    latent_storage['img_ids'] = img_ids.cpu()
    
    # this is ignored for schnell
    inject_list = [False] * len(timesteps[:-1])
    inject_list[config.start_inject_step:config.end_inject_step] = [True] * (config.end_inject_step - config.start_inject_step)
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    ref_guidance_vec = torch.full((img.shape[0],), 1., device=img.device, dtype=img.dtype)

    next_step_velocity = None
    ref_next_step_velocitys = None
    for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), 'sampling'):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        t = t_prev if inverse else t_curr
        
        if not inverse and ref_imgs is not None:
            t_str = f't_{t:.4f}'
            
            if t_str in latent_storage:
                stored_refs = latent_storage[t_str]
                if stored_refs is not None and len(stored_refs) == len(ref_imgs):
                    ref_imgs = [ref.to(ref_imgs[0].device) for ref in stored_refs]
                del latent_storage[t_str]

        if next_step_velocity is None:
            pred, ref_preds = model(
                img=img,
                ref_imgs=ref_imgs,
                ref_masks=ref_masks,
                img_ids=img_ids,
                txt=txt,
                ref_txts=ref_txts,
                txt_ids=txt_ids,
                y=vec,
                ref_ys=ref_vecs,
                timesteps=t_vec,
                guidance=guidance_vec,
                ref_guidance=ref_guidance_vec,
                t=t,
                inverse=inverse,
                order=1,
                inject=inject_list[i],
                config=config,
                latent_storage=latent_storage
            )
        else:
            pred = next_step_velocity
            ref_preds = ref_next_step_velocitys
        
        img_mid = img + (t_prev - t_curr) / 2 * pred
        
        if ref_preds is not None and len(ref_preds) > 0:
            ref_img_mids = [ref_img + (t_prev - t_curr) / 2 * ref_pred  for ref_img, ref_pred in zip(ref_imgs, ref_preds)]
        else:
            ref_img_mids = None

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        pred_mid, ref_pred_mids = model(
            img=img_mid,
            ref_imgs=ref_img_mids,
            ref_masks=ref_masks,
            img_ids=img_ids,
            txt=txt,
            ref_txts=ref_txts,
            txt_ids=txt_ids,
            y=vec,
            ref_ys=ref_vecs,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            ref_guidance=ref_guidance_vec,
            t=t,
            inverse=inverse,
            order=2,
            inject=inject_list[i],
            config=config,
            latent_storage=latent_storage
        )
        next_step_velocity = pred_mid
        if ref_pred_mids is not None:
            ref_next_step_velocitys = ref_pred_mids
        
        img = img + (t_prev - t_curr) * pred_mid
        if ref_pred_mids is not None:
            ref_imgs = [ref_img + (t_prev - t_curr) * ref_pred_mid for ref_img, ref_pred_mid in zip(ref_imgs, ref_pred_mids)]

        if inverse and store_latents:
            t_str = f't_{t:.4f}'
            latent_storage[t_str] = img.cpu().unsqueeze(0)
            
        if callback is not None:
            callback(i+1, config.num_steps*3)
            
    if ref_imgs is not None:
        return img, ref_imgs
    else:
        return img


def denoise_midpoint(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )
        next_step_velocity = pred_mid
        
        img = img + (t_prev - t_curr) * pred_mid

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
