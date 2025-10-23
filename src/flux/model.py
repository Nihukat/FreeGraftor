from dataclasses import dataclass
from typing import List
import torch
from torch import Tensor, nn

from flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

@torch.compile
class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        ref_imgs: List[Tensor] = None, 
        ref_masks: List[Tensor] = None,
        ref_txts: List[Tensor] = None,
        ref_ys: List[Tensor] = None,
        ref_guidance: Tensor | None = None,
        t: float = 0.0,
        inverse: bool = False,
        order: int = 1,
        inject: bool = False,
        config = None,  # GenerationConfig
        latent_storage: dict = None,  # For img_ids storage
    ) -> List[Tensor]:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        
        ref_vecs = None
        if ref_imgs is not None and ref_txts is not None and ref_ys is not None:
            ref_imgs_, ref_txts_ = [], []
            ref_vecs = []
            for ref_img, ref_txt, ref_y in zip(ref_imgs, ref_txts, ref_ys):
                ref_img = self.img_in(ref_img)
                ref_vec = self.time_in(timestep_embedding(timesteps, 256))
                if self.params.guidance_embed:
                    if ref_guidance is None:
                        raise ValueError("Didn't get guidance strength for guidance distilled model.")
                    ref_vec = ref_vec + self.guidance_in(timestep_embedding(ref_guidance, 256))
                ref_vec = ref_vec + self.vector_in(ref_y)
                ref_txt = self.txt_in(ref_txt)
                ref_imgs_.append(ref_img)
                ref_txts_.append(ref_txt)
                ref_vecs.append(ref_vec)
            ref_imgs = ref_imgs_
            ref_txts = ref_txts_
        
        context = {
            'pe_embedder': self.pe_embedder,
            't': t,
            'inverse': inverse,
            'order': order,
            'inject': inject,
        }
        
        if config is not None:
            context.update({
                'height': config.height,
                'width': config.width,
                'sim_threshold': config.sim_threshold,
                'cyc_threshold': config.cyc_threshold,
                'inject_block_ids': config.inject_block_ids,
                'inject_match_dropout': config.inject_match_dropout,
            })
        
        if latent_storage is not None and 'img_ids' in latent_storage:
            context['img_ids'] = latent_storage['img_ids']
        
        txt_pe = self.pe_embedder(txt_ids)
        img_pe = self.pe_embedder(img_ids)

        cnt = 0

        context['block_type'] = 'double'
        for block in self.double_blocks:
            context['global_block_id'] = cnt
            img, ref_imgs, txt, ref_txts = block(img=img, ref_imgs=ref_imgs, ref_masks=ref_masks, txt=txt, ref_txts=ref_txts, 
                             vec=vec, ref_vecs=ref_vecs, txt_pe=txt_pe, img_pe=img_pe, context=context)
            cnt += 1
        
        context['block_type'] = 'single'
        for block in self.single_blocks:
            context['global_block_id'] = cnt
            img, ref_imgs, txt, ref_txts = block(img=img, ref_imgs=ref_imgs, ref_masks=ref_masks, txt=txt, ref_txts=ref_txts, 
                             vec=vec, ref_vecs=ref_vecs, txt_pe=txt_pe, img_pe=img_pe, context=context)
            cnt += 1

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    
        if ref_imgs is not None:
            ref_imgs_ = []
            for ref_img, ref_vec in zip(ref_imgs, ref_vecs):
                ref_img = self.final_layer(ref_img, ref_vec) # (N, T, T
                ref_imgs_.append(ref_img)
            ref_imgs = ref_imgs_
            
        return img, ref_imgs
