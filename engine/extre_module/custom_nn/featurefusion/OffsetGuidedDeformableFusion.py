"""Offset-Guided Deformable Fusion (v6)"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.core import register
from engine.deim.dfine_decoder import MSDeformableAttention
from engine.deim.utils import deformable_attention_core_func_v2

class GlobalOffsetPredictor(nn.Module):
    """从多尺度特征联合预测全局偏移 (dx, dy)。"""

    def __init__(self, in_channels_deepest, max_offset=30.0):
        """Args:"""
        super().__init__()
        self.max_offset = max_offset
        c = in_channels_deepest
        self.offset_net = nn.Sequential(nn.Conv2d(c * 2, c // 2, kernel_size=3, padding=1), nn.BatchNorm2d(c // 2), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c // 2, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        nn.init.constant_(self.offset_net[-1].weight, 0)
        nn.init.constant_(self.offset_net[-1].bias, 0)

    def forward(self, x_rgb_deep, x_ir_deep):
        """Args:"""
        cat = torch.cat([x_rgb_deep, x_ir_deep], dim=1)
        offset = self.offset_net(cat)
        offset = torch.tanh(offset) * self.max_offset
        return offset

def shift_feature(feat, offset_pixels, stride):
    """用 grid_sample 对特征图做亚像素级全局平移。"""
    b, c, h, w = feat.shape
    dx_norm = offset_pixels[:, 0:1] / stride * (2.0 / w)
    dy_norm = offset_pixels[:, 1:2] / stride * (2.0 / h)
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype), torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    shift = torch.cat([dx_norm, dy_norm], dim=1)
    grid = grid - shift.reshape(b, 1, 1, 2)
    shifted = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return shifted

class OffsetGuidedDeformableFusionPerLevel(nn.Module):
    """单层偏移引导可变形融合。"""

    def __init__(self, dim, n_points=8, n_heads=8):
        super().__init__()
        self.dim = dim
        self.offset_predictor = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True), nn.Conv2d(dim, n_heads * n_points * 2, kernel_size=1))
        nn.init.constant_(self.offset_predictor[-1].weight, 0)
        nn.init.constant_(self.offset_predictor[-1].bias, 0)
        self.attn_weights = nn.Linear(dim, n_heads * n_points)
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = dim // n_heads
        self.fusion = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_rgb, x_ir, offset_pixels, stride):
        """Args:"""
        b, c, h, w = x_rgb.shape
        x_ir_shifted = shift_feature(x_ir, offset_pixels, stride)
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=x_rgb.device), torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=x_rgb.device), indexing='ij')
        ref_y = ref_y.reshape(-1) / h
        ref_x = ref_x.reshape(-1) / w
        ref = torch.stack((ref_x, ref_y), -1)
        ref = ref[None, :, None, :].repeat(b, 1, 1, 1)
        spatial_shapes = torch.tensor([(h, w)], device=x_rgb.device)
        cat_feat = torch.cat([x_rgb, x_ir_shifted], dim=1)
        offsets = self.offset_predictor(cat_feat)
        offsets = offsets.flatten(2).transpose(1, 2)
        offsets = offsets.view(b, h * w, self.n_heads, 1, self.n_points, 2)
        query = x_rgb.flatten(2).transpose(1, 2)
        attn_w = self.attn_weights(query)
        attn_w = attn_w.view(b, h * w, self.n_heads, self.n_points)
        attn_w = F.softmax(attn_w, dim=-1)
        ref_reshaped = ref.reshape(b, h * w, 1, 1, 1, 2)
        offset_normalizer = spatial_shapes.flip([1]).reshape(1, 1, 1, 1, 1, 2)
        sampling_locations = ref_reshaped + offsets / offset_normalizer
        sampling_locations = sampling_locations.flatten(3, 4)
        value = x_ir_shifted.flatten(2).transpose(1, 2).view(b, h * w, self.n_heads, self.head_dim)
        x_ir_aligned = deformable_attention_core_func_v2(value=value, value_spatial_shapes=spatial_shapes, sampling_locations=sampling_locations, attention_weights=attn_w, num_points_list=[self.n_points], method='default', value_shape='reshape')
        x_ir_aligned = x_ir_aligned.transpose(1, 2).reshape(b, c, h, w)
        fused = self.fusion(torch.cat([x_rgb, x_ir_aligned], dim=1))
        out = x_rgb + self.alpha * (fused - x_rgb)
        return out

@register()
class OffsetGuidedDeformableFusionList(nn.Module):
    """多尺度偏移引导可变形融合。"""

    def __init__(self, in_channels=[256, 512, 1024], strides=[8, 16, 32], n_points=8, n_heads=8, max_offset=30.0, offset_ema_alpha=0.0):
        super().__init__()
        assert len(in_channels) == len(strides)
        self.strides = strides
        self.offset_predictor = GlobalOffsetPredictor(in_channels_deepest=in_channels[-1], max_offset=max_offset)
        self.fusions = nn.ModuleList([OffsetGuidedDeformableFusionPerLevel(c, n_points=n_points, n_heads=n_heads) for c in in_channels])
        self.offset_ema_alpha = offset_ema_alpha
        self._prev_offset = None

    def forward(self, feat_rgb, feat_ir):
        """Args:"""
        assert len(feat_rgb) == len(feat_ir) == len(self.fusions)
        offset_pixels = self.offset_predictor(feat_rgb[-1], feat_ir[-1])
        if not self.training and self.offset_ema_alpha > 0:
            alpha = self.offset_ema_alpha
            if self._prev_offset is not None:
                if offset_pixels.shape[0] == 1 and self._prev_offset.shape[0] == 1:
                    offset_pixels = alpha * self._prev_offset.detach() + (1 - alpha) * offset_pixels
            self._prev_offset = offset_pixels.detach().clone()
        outputs = []
        for i, (f_rgb, f_ir) in enumerate(zip(feat_rgb, feat_ir)):
            out = self.fusions[i](f_rgb, f_ir, offset_pixels, self.strides[i])
            outputs.append(out)
        return outputs
