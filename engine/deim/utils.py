"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)  
Copyright (c) 2023 . All Rights Reserved.    
"""
import math
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def inverse_sigmoid(x: torch.Tensor, eps: float=1e-05) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))

def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init

def deformable_attention_core_func_v2(value: torch.Tensor, value_spatial_shapes, sampling_locations: torch.Tensor, attention_weights: torch.Tensor, num_points_list: List[int], method='default', value_shape='default'):
    """Args:"""
    if value_shape == 'default':
        bs, n_head, c, _ = value[0].shape
    elif value_shape == 'reshape':
        bs, _, n_head, c = value.shape
        split_shape = [h * w for h, w in value_spatial_shapes]
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    _, Len_q, _, _, _ = sampling_locations.shape
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1
    elif method == 'discrete':
        sampling_grids = sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]
        if method == 'default':
            sampling_value_l = F.grid_sample(value_l, sampling_grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
        elif method == 'discrete':
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)
            s_idx = torch.arange(sampling_coord.shape[0], device=value_l.device).unsqueeze(-1).repeat(1, sampling_coord.shape[1])
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])
        sampling_value_list.append(sampling_value_l)
    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)

def get_activation(act: str, inpace: bool=True):
    """get activation"""
    if act is None:
        return nn.Identity()
    elif isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'silu':
        m = nn.SiLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError('')
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m
