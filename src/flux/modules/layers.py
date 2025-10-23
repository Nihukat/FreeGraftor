import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from typing import List
import copy

from flux.math import attention, rope

from core.semantic_match import get_match, apply_match

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(q), k.to(q)

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, ref_imgs: List[Tensor], ref_masks: List[Tensor], txt: Tensor, ref_txts: List[Tensor], 
                vec: Tensor, ref_vecs: List[Tensor], txt_pe: Tensor, img_pe: Tensor, context) -> tuple[Tensor, Tensor]:
        requires_inject = context['inject'] and context['global_block_id'] in context['inject_block_ids'] and not context['inverse'] and ref_imgs is not None
        ref_imgs_, ref_txts_ = None, None
        if ref_imgs is not None and ref_txts is not None and ref_vecs is not None:
            ref_img_ks, ref_img_vs = [], []
            ref_imgs_, ref_txts_ = [], []
            ref_img_modulateds = []
            for ref_img, ref_txt, ref_vec in zip(ref_imgs, ref_txts, ref_vecs):
                ref_img_mod1, ref_img_mod2 = self.img_mod(ref_vec)
                ref_txt_mod1, ref_txt_mod2 = self.txt_mod(ref_vec) 
                
                ref_img_modulated = self.img_norm1(ref_img)
                ref_img_modulated = (1 + ref_img_mod1.scale) * ref_img_modulated + ref_img_mod1.shift
                ref_img_modulateds.append(ref_img_modulated)
                ref_img_qkv = self.img_attn.qkv(ref_img_modulated)
                ref_img_q, ref_img_k, ref_img_v = rearrange(ref_img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
                ref_img_q, ref_img_k = self.img_attn.norm(ref_img_q, ref_img_k)   
                
                ref_img_ks.append(ref_img_k)
                ref_img_vs.append(ref_img_v)
                
                ref_txt_modulated = self.txt_norm1(ref_txt)
                ref_txt_modulated = (1 + ref_txt_mod1.scale) * ref_txt_modulated + ref_txt_mod1.shift
                ref_txt_qkv = self.txt_attn.qkv(ref_txt_modulated)
                ref_txt_q, ref_txt_k, ref_txt_v = rearrange(ref_txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
                ref_txt_q, ref_txt_k = self.txt_attn.norm(ref_txt_q, ref_txt_k)   
                
                ref_q = torch.cat((ref_txt_q, ref_img_q), dim=2)
                ref_k = torch.cat((ref_txt_k, ref_img_k), dim=2)
                ref_v = torch.cat((ref_txt_v, ref_img_v), dim=2)    
                pe = torch.cat((txt_pe, img_pe), dim=2)
                
                ref_attn = attention(ref_q, ref_k, ref_v, pe=pe)
                ref_txt_attn, ref_img_attn = ref_attn[:, : txt.shape[1]], ref_attn[:, txt.shape[1] :]      
                
                ref_img = ref_img + ref_img_mod1.gate * self.img_attn.proj(ref_img_attn)
                ref_img_mlp_out = self.img_mlp((1 + ref_img_mod2.scale) * self.img_norm2(ref_img) + ref_img_mod2.shift)   
                ref_img = ref_img + ref_img_mod2.gate * ref_img_mlp_out  
                ref_imgs_.append(ref_img)       
                
                ref_txt = ref_txt + ref_txt_mod1.gate * self.txt_attn.proj(ref_txt_attn)
                ref_txt_mlp_out = self.txt_mlp((1 + ref_txt_mod2.scale) * self.txt_norm2(ref_txt) + ref_txt_mod2.shift)
                ref_txt = ref_txt + ref_txt_mod2.gate * ref_txt_mlp_out  
                ref_txts_.append(ref_txt)
        
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        pe = torch.cat((txt_pe, img_pe), dim=2)
            
        if requires_inject:
            for ref_id, ref_img_modulated in enumerate(ref_img_modulateds):
                src2tar_2d_xy, src2tar_match_mask, src2tar_1d = get_match(ref_img_modulated, img_modulated, 
                                                                          h_src=context['height'] // 16, w_src=context['width'] // 16,
                                                                          h_tar=context['height'] // 16, w_tar=context['width'] // 16,
                                                                          sim_threshold=context['sim_threshold'], cyc_threshold=context['cyc_threshold'])
                
                img_ids = context['img_ids'].to(img)
                    
                ref_img_ids = apply_match(src2tar_2d_xy, img_ids, h_tar=context['height'] // 16, w_tar=context['width'] // 16)
                ref_img_pe = context['pe_embedder'](ref_img_ids)
                
                ref_mask = ref_masks[ref_id].to(img)
                ref_mask *= src2tar_match_mask.squeeze(dim=0)
                
                dropout_p = context['t'] * context['inject_match_dropout']
                ref_mask = dropout_binary_mask(ref_mask, dropout_p)
                            
                seg_img_k = ref_img_ks[ref_id][:,:,ref_mask>0]
                seg_img_v = ref_img_vs[ref_id][:,:,ref_mask>0]
                seg_img_pe = ref_img_pe[:,:,ref_mask>0]

                k = torch.cat((k, seg_img_k), dim=2)
                v = torch.cat((v, seg_img_v), dim=2)
                pe = torch.cat((pe, seg_img_pe), dim=2)

        attn = attention(q, k, v, pe=pe)    
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        
        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img_mlp_out = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        img = img + img_mod2.gate * img_mlp_out

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt_mlp_out = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = txt + txt_mod2.gate * txt_mlp_out
        
        return img, ref_imgs_, txt, ref_txts_

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, img: Tensor, ref_imgs: List[Tensor], ref_masks: List[Tensor], txt: Tensor, ref_txts: List[Tensor], 
                vec: Tensor, ref_vecs: List[Tensor], txt_pe: Tensor, img_pe: Tensor, context) -> Tensor:
        requires_inject = context['inject'] and context['global_block_id'] in context['inject_block_ids'] and not context['inverse'] and ref_imgs is not None
        
        l1 = txt.shape[1]
        l2 = img.shape[1]
            
        pe = torch.cat((txt_pe, img_pe), dim=2)
        
        ref_imgs_, ref_txts_ = None, None
        if ref_imgs is not None and ref_txts is not None and ref_vecs is not None:
            ref_imgs_, ref_txts_ = [], []
            ref_img_ks, ref_img_vs = [], []
            ref_img_mods = []
            for ref_img, ref_txt, ref_vec in zip(ref_imgs, ref_txts, ref_vecs):
                ref_mod, _ = self.modulation(ref_vec)
                
                ref_x = torch.cat((ref_txt, ref_img), 1)
                ref_x_mod = (1 + ref_mod.scale) * self.pre_norm(ref_x) + ref_mod.shift
                ref_txt_mod, ref_img_mod = torch.split(ref_x_mod, [l1, l2], dim=1)
                ref_img_mods.append(ref_img_mod)
                ref_qkv, ref_mlp = torch.split(self.linear1(ref_x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
                ref_q, ref_k, ref_v = rearrange(ref_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
                ref_q, ref_k = self.norm(ref_q, ref_k)
                ref_txt_k, ref_img_k = ref_k[:,:,:l1], ref_k[:,:,l1:]
                ref_txt_v, ref_img_v = ref_v[:,:,:l1], ref_v[:,:,l1:]     
                ref_img_ks.append(ref_img_k)
                ref_img_vs.append(ref_img_v)
                
                ref_attn = attention(ref_q, ref_k, ref_v, pe=pe)
                ref_mlp_out = self.mlp_act(ref_mlp)
                
                ref_output = self.linear2(torch.cat((ref_attn, ref_mlp_out), 2))
                ref_output_ = ref_x + ref_mod.gate * ref_output                
                
                ref_txt = ref_output_[:,:l1]
                ref_img = ref_output_[:,l1:l1+l2]   
                
                ref_imgs_.append(ref_img)
                ref_txts_.append(ref_txt)
        
        mod, _ = self.modulation(vec)
        x = torch.cat((txt, img), 1)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        
        x_mod_copy = copy.deepcopy(x_mod)
        txt_mod, img_mod = torch.split(x_mod, [l1, l2], dim=1)
        qkv, _ = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        
        x_mod = x_mod_copy
        _, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k)
        txt_k, img_k = k[:,:,:l1], k[:,:,l1:]
        txt_v, img_v = v[:,:,:l1], v[:,:,l1:]
        if requires_inject:
            for ref_id, ref_img in enumerate(ref_imgs):
                ref_img_mod = ref_img_mods[ref_id]
                src2tar_2d_xy, src2tar_match_mask, src2tar_1d = get_match(ref_img_mod, img_mod, 
                                                                          h_src=context['height'] // 16, w_src=context['width'] // 16,
                                                                          h_tar=context['height'] // 16, w_tar=context['width'] // 16,
                                                                          sim_threshold=context['sim_threshold'], cyc_threshold=context['cyc_threshold'])
                
                img_ids = context['img_ids'].to(img)
                    
                ref_img_ids = apply_match(src2tar_2d_xy, img_ids, h_tar=context['height'] // 16, w_tar=context['width'] // 16)
                ref_img_pe = context['pe_embedder'](ref_img_ids)            
                
                ref_mask = ref_masks[ref_id].to(img)
                ref_mask *= src2tar_match_mask.squeeze(dim=0)
                
                dropout_p = context['t'] * context['inject_match_dropout']
                
                ref_mask = dropout_binary_mask(ref_mask, dropout_p)
                
                seg_img_k = ref_img_ks[ref_id][:,:,ref_mask>0]
                seg_img_v = ref_img_vs[ref_id][:,:,ref_mask>0]
                img_k = torch.cat([img_k, seg_img_k], dim=2)
                img_v = torch.cat([img_v, seg_img_v], dim=2)
                seg_img_pe = ref_img_pe[:,:,ref_mask>0]
                
                pe = torch.cat((pe, seg_img_pe), dim=2)
                k = torch.cat([k, seg_img_k], dim=2)
                v = torch.cat([v, seg_img_v], dim=2)
        
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        mlp_out = self.mlp_act(mlp)
        
        output = self.linear2(torch.cat((attn, mlp_out), 2))
        output_ = x + mod.gate * output
        
        txt = output_[:,:l1]
        img = output_[:,l1:l1+l2]
        
        return img, ref_imgs_, txt, ref_txts_

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

def dropout_binary_mask(binary_mask, p):
    assert 0 <= p <= 1
    mask = torch.bernoulli((1 - p) * binary_mask) 
    return mask
