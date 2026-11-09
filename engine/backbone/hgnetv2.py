"""
reference     
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re
from .common import FrozenBatchNorm2d
from ..core import register
from ..extre_module.custom_nn.attention.ema import EMA
from ..extre_module.custom_nn.attention.simam import SimAM
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized
from ..logger_module import get_logger
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_
logger = get_logger(__name__)
RED, GREEN, RESET = ('\x1b[91m', '\x1b[92m', '\x1b[0m')
__all__ = ['HGNetv2']

class LearnableAffineBlock(nn.Module):
    """可学习的仿射变换模块 (Learnable Affine Block)"""

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        """前向传播：执行仿射变换"""
        return self.scale * x + self.bias

class ConvBNAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1, padding='', use_act=True, use_lab=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False))
        else:
            self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x

class LightConvBNAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, groups=1, use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs, use_act=True, use_lab=use_lab)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class StemBlock(nn.Module):
    """HGNetv2 的 Stem Block，作为网络的输入处理模块。"""

    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)

    def forward(self, x):
        """前向传播过程："""
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class EseModule(nn.Module):

    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)

class HG_Block(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False, use_lab=False, agg='ese', drop_path=0.0):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(LightConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size, use_lab=use_lab))
            else:
                self.layers.append(ConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size, stride=1, use_lab=use_lab))
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab)
            aggregation_excitation_conv = ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            self.aggregation = nn.Sequential(aggregation_squeeze_conv, aggregation_excitation_conv)
        elif agg == 'ese':
            aggregation_conv = ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'ema':
            aggregation_conv = ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            att = EMA(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'simam':
            aggregation_conv = ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            att = SimAM()
            self.aggregation = nn.Sequential(aggregation_conv, att)
        else:
            raise Exception(f'param agg{agg} Illegal.')
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x

class HG_Stage(nn.Module):
    """HG_Stage 模块 (Hourglass Stage)"""

    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0.0):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs, use_act=False, use_lab=use_lab)
        else:
            self.downsample = nn.Identity()
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(HG_Block(in_chs if i == 0 else out_chs, mid_chs, out_chs, layer_num, residual=False if i == 0 else True, kernel_size=kernel_size, light_block=light_block, use_lab=use_lab, agg=agg, drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        """前向传播过程："""
        x = self.downsample(x)
        x = self.blocks(x)
        return x

@register()
class HGNetv2(nn.Module):
    """HGNetV2"""
    arch_configs = {'Atto': {'stem_channels': [3, 16, 16], 'stage_config': {'stage1': [16, 16, 64, 1, False, False, 3, 3], 'stage2': [64, 32, 256, 1, True, False, 3, 3], 'stage3': [256, 64, 256, 1, True, True, 3, 3]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'}, 'Nano': {'stem_channels': [3, 16, 16], 'stage_config': {'stage1': [16, 16, 64, 1, False, False, 3, 3], 'stage2': [64, 32, 256, 1, True, False, 3, 3], 'stage3': [256, 64, 512, 2, True, True, 5, 3], 'stage4': [512, 128, 1024, 1, True, True, 5, 3]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'}, 'Femto': {'stem_channels': [3, 16, 16], 'stage_config': {'stage1': [16, 16, 64, 1, False, False, 3, 3], 'stage2': [64, 32, 256, 1, True, False, 3, 3], 'stage3': [256, 64, 512, 1, True, True, 5, 3]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'}, 'Pico': {'stem_channels': [3, 16, 16], 'stage_config': {'stage1': [16, 16, 64, 1, False, False, 3, 3], 'stage2': [64, 32, 256, 1, True, False, 3, 3], 'stage3': [256, 64, 512, 2, True, True, 5, 3]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'}, 'B0': {'stem_channels': [3, 16, 16], 'stage_config': {'stage1': [16, 16, 64, 1, False, False, 3, 3], 'stage2': [64, 32, 256, 1, True, False, 3, 3], 'stage3': [256, 64, 512, 2, True, True, 5, 3], 'stage4': [512, 128, 1024, 1, True, True, 5, 3]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'}, 'B1': {'stem_channels': [3, 24, 32], 'stage_config': {'stage1': [32, 32, 64, 1, False, False, 3, 3], 'stage2': [64, 48, 256, 1, True, False, 3, 3], 'stage3': [256, 96, 512, 2, True, True, 5, 3], 'stage4': [512, 192, 1024, 1, True, True, 5, 3]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'}, 'B2': {'stem_channels': [3, 24, 32], 'stage_config': {'stage1': [32, 32, 96, 1, False, False, 3, 4], 'stage2': [96, 64, 384, 1, True, False, 3, 4], 'stage3': [384, 128, 768, 3, True, True, 5, 4], 'stage4': [768, 256, 1536, 1, True, True, 5, 4]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'}, 'B3': {'stem_channels': [3, 24, 32], 'stage_config': {'stage1': [32, 32, 128, 1, False, False, 3, 5], 'stage2': [128, 64, 512, 1, True, False, 3, 5], 'stage3': [512, 128, 1024, 3, True, True, 5, 5], 'stage4': [1024, 256, 2048, 1, True, True, 5, 5]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'}, 'B4': {'stem_channels': [3, 32, 48], 'stage_config': {'stage1': [48, 48, 128, 1, False, False, 3, 6], 'stage2': [128, 96, 512, 1, True, False, 3, 6], 'stage3': [512, 192, 1024, 3, True, True, 5, 6], 'stage4': [1024, 384, 2048, 1, True, True, 5, 6]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'}, 'B5': {'stem_channels': [3, 32, 64], 'stage_config': {'stage1': [64, 64, 128, 1, False, False, 3, 6], 'stage2': [128, 128, 512, 2, True, False, 3, 6], 'stage3': [512, 256, 1024, 5, True, True, 5, 6], 'stage4': [1024, 512, 2048, 2, True, True, 5, 6]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'}, 'B6': {'stem_channels': [3, 48, 96], 'stage_config': {'stage1': [96, 96, 192, 2, False, False, 3, 6], 'stage2': [192, 192, 512, 3, True, False, 3, 6], 'stage3': [512, 384, 1024, 6, True, True, 5, 6], 'stage4': [1024, 768, 2048, 3, True, True, 5, 6]}, 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'}}

    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3], freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', in_channels=3, local_model_dir='weight/hgnetv2/'):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx
        stem_channels = list(self.arch_configs[name]['stem_channels'])
        stem_channels[0] = in_channels
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']
        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]
        self.stem = StemBlock(in_chs=stem_channels[0], mid_chs=stem_channels[1], out_chs=stem_channels[2], use_lab=use_lab)
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]
            self.stages.append(HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg))
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])
        if freeze_norm:
            self._freeze_norm(self)
        if pretrained:
            try:
                if name in ['Atto', 'Femto', 'Pico']:
                    model_path = local_model_dir + 'PPHGNetV2_' + 'B0' + '_stage1.pth'
                else:
                    model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    logger.info(f'Loaded stage1 {name} HGNetV2 from local file.')
                else:
                    if is_dist_available_and_initialized() and torch.distributed.get_rank() == 0:
                        logger.warning(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        logger.warning(GREEN + 'Please check your network connection. Or download the model manually from ' + RESET + f'{download_url}' + GREEN + ' to ' + RESET + f'{local_model_dir}.' + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                        Multiprocess_sync()
                    else:
                        Multiprocess_sync()
                        state = torch.load(local_model_dir)
                    logger.info(f'Loaded stage1 {name} HGNetV2 from URL.')
                if name == 'Atto':
                    self.load_partial_state_dict(self, state)
                else:
                    logger.info(RED + f'Loading Pretrained State Dict Key Names:{state.keys()}' + RESET)
                    self.load_state_dict(state, strict=False)
            except (Exception, KeyboardInterrupt) as e:
                if is_dist_available_and_initialized() and torch.distributed.get_rank() == 0 or not is_dist_available_and_initialized():
                    print(f'{str(e)}')
                    logger.error(RED + 'CRITICAL WARNING: Failed to load pretrained HGNetV2 model' + RESET)
                    logger.error(GREEN + 'Please check your network connection. Or download the model manually from ' + RESET + f'{download_url}' + GREEN + ' to ' + RESET + f'{local_model_dir}.' + RESET)
                exit()

    @staticmethod
    def load_partial_state_dict(model, state_dict):
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        logger.info(RED + f'Loading Pretrained State Dict Key Names:{filtered_dict.keys()}' + RESET)
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
