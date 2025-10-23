import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def get_match(src_features, tar_features, h_src=64, w_src=64, h_tar=64, w_tar=64, cyc_threshold=1.5, tar_mask=None, sim_threshold=0.2):
    src2tar_1d, sim_mask_1d = match_feature_src2tar(src_features, tar_features, tar_mask=tar_mask, sim_threshold=sim_threshold)
    src2tar_flow, src2src_2d_xy = mapping_to_flow(src2tar_1d, h_src, w_src, h_tar, w_tar)
    src2tar_flow = src2tar_flow.to(src_features)
    src2src_2d_xy = src2src_2d_xy.to(src_features)
    src2tar_2d_xy = src2src_2d_xy + src2tar_flow
    
    if cyc_threshold > 0:
        tar2src_1d, _ = match_feature_src2tar(tar_features, src_features)
        tar2src_flow, tar2tar_2d_xy = mapping_to_flow(tar2src_1d, h_tar, w_tar, h_src, w_src)
        tar2src_flow = tar2src_flow.to(src_features)
        src_tar2src_flow = F.grid_sample(rearrange(tar2src_flow, 'b h w c -> b c h w'), src2tar_2d_xy, align_corners=False)
        flow_bias = rearrange(src2tar_flow, 'b h w c -> b c h w') + src_tar2src_flow
        bias_distance = torch.norm(flow_bias, dim=1, p=2)
        bias_mask_2d = (bias_distance < cyc_threshold).to(src_features)
        bias_mask_1d = rearrange(bias_mask_2d, 'b h w -> b (h w)', h=h_src, w=w_src)
    else:
        bias_mask_1d = None
        
    mask_1d = bias_mask_1d * sim_mask_1d if bias_mask_1d is not None else sim_mask_1d
    
    return src2tar_2d_xy, mask_1d, src2tar_1d

def apply_match(src2tar_2d_xy, tar_features, h_tar=64, w_tar=64):
    tar_features_ = rearrange(tar_features, 'b (h w) c -> b c h w', h=h_tar, w=w_tar)
    new_features_ = F.grid_sample(tar_features_, src2tar_2d_xy, align_corners=False)
    new_features = rearrange(new_features_, 'b c h w -> b (h w) c')
    return new_features

def mapping_to_flow(src2tar_1d, h_src=64, w_src=64, h_tar=64, w_tar=64):
    src2tar_2d = rearrange(src2tar_1d, 'b (hs ws) -> b hs ws', hs=h_src, ws=w_src)
    
    src2tar_2d_y = src2tar_2d // w_tar
    src2tar_2d_x = src2tar_2d % w_tar
    src2tar_2d_y_normed = (src2tar_2d_y.float() / h_tar) * 2 - 1 + 1 / h_tar
    src2tar_2d_x_normed = (src2tar_2d_x.float() / w_tar) * 2 - 1 + 1 / w_tar
    src2tar_2d_xy = torch.stack([src2tar_2d_x_normed, src2tar_2d_y_normed], dim=-1)
    
    device = src2tar_2d.device
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h_src, device=device, dtype=torch.float32),
        torch.arange(w_src, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    src2src_2d_y_normed = (y_grid / h_src) * 2 - 1 + 1 / h_src
    src2src_2d_x_normed = (x_grid / w_src) * 2 - 1 + 1 / w_src
    src2src_2d_xy = torch.stack([src2src_2d_x_normed, src2src_2d_y_normed], dim=-1).unsqueeze(0)
    
    src2tar_flow = src2tar_2d_xy - src2src_2d_xy
    smoothed_src2tar_flow = smooth_flow(src2tar_flow)
    return smoothed_src2tar_flow, src2src_2d_xy

def smooth_flow(flow):
    return flow

def match_feature_src2tar(src_features, tar_features, tar_mask=None, sim_threshold=0.5):
    B, M, D = src_features.shape
    B, N, D = tar_features.shape
    
    chunk_size = 1024 if M > 1024 else M
    
    max_ids = torch.zeros(B, M, dtype=torch.long, device=src_features.device)
    max_sims = torch.zeros(B, M, device=src_features.device)
    
    src_norm = torch.norm(src_features, p=2, dim=-1, keepdim=True)
    tar_norm = torch.norm(tar_features, p=2, dim=-1, keepdim=True)
    
    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        src_chunk = src_features[:, i:end_i]
        src_norm_chunk = src_norm[:, i:end_i]
        
        similarity = torch.bmm(src_chunk, tar_features.transpose(1, 2))
        norm_product = torch.bmm(src_norm_chunk, tar_norm.transpose(1, 2))
        sim_chunk = similarity / (norm_product + 1e-10)
        
        if tar_mask is not None:
            mask_expanded = tar_mask.unsqueeze(1).expand(-1, sim_chunk.size(1), -1)
            sim_chunk = sim_chunk.masked_fill(~mask_expanded, float('-inf'))
        
        chunk_max_sims, chunk_max_ids = sim_chunk.max(dim=2)
        max_ids[:, i:end_i] = chunk_max_ids
        max_sims[:, i:end_i] = chunk_max_sims
    
    sim_mask = (max_sims > sim_threshold).to(src_features)
    return max_ids, sim_mask