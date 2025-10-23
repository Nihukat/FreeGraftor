import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=q.device.type == 'cuda'):
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=False,
        )
    
    B, H, L, D = x.shape
    x = x.transpose(1, 2).reshape(B, L, H * D)

    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    
    device = pos.device
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    omega = 1.0 / (theta ** scale)
    
    out = torch.einsum("...n,d->...nd", pos, omega)
    
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    
    out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    
    b, n, d = out.shape[:-1]
    out = out.view(b, n, d, 2, 2)
    
    return out.to(dtype=torch.float32)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis_k: Tensor, freqs_cis_q: Tensor = None) -> tuple[Tensor, Tensor]:
    if freqs_cis_q is None:
        freqs_cis_q = freqs_cis_k[:, :, :xq.shape[2]]
    
    original_dtype = xq.dtype
    
    q_shape = xq.shape
    k_shape = xk.shape
    
    xq_ = xq.view(*q_shape[:-1], -1, 1, 2)
    xk_ = xk.view(*k_shape[:-1], -1, 1, 2)
    
    xq_out = freqs_cis_q[..., 0] * xq_[..., 0] + freqs_cis_q[..., 1] * xq_[..., 1]
    xk_out = freqs_cis_k[..., 0] * xk_[..., 0] + freqs_cis_k[..., 1] * xk_[..., 1]
    
    return (xq_out.view(q_shape).to(original_dtype), 
            xk_out.view(k_shape).to(original_dtype))
