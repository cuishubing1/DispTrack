"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
--------------------------------------------------------------------------------- 
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""
import math
import copy
import functools
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List
from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from ..core import register
__all__ = ['DFINETransformer']

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList((nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MSDeformableAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, method='default', offset_scale=0.5):
        """Multi-Scale Deformable Attention"""
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]
        self.num_points_list = num_points_list
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))
        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)
        self._reset_parameters()
        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, value: torch.Tensor, value_spatial_shapes: List[int]):
        """Args:"""
        bs, Len_q = query.shape[:2]
        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_offsets = sampling_offsets.view(bs, Len_q, self.num_heads, self.num_levels, self.num_points_list[0], 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
            sampling_locations = sampling_locations.flatten(3, 4)
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)
        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, activation='relu', n_levels=4, n_points=4, cross_attn_method='default', layer_scale=None, use_ir_auxiliary=False):
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.use_ir_auxiliary = use_ir_auxiliary
        if use_ir_auxiliary:
            self.ir_cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
            self.ir_dropout = nn.Dropout(dropout)
            self.ir_gateway = Gate(d_model)
            self.ir_alpha = nn.Parameter(torch.zeros(1))
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None, ir_value=None, ir_spatial_shapes=None):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)
        target = self.gateway(target, self.dropout2(target2))
        if self.use_ir_auxiliary and ir_value is not None:
            ir_target2 = self.ir_cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, ir_value, ir_spatial_shapes if ir_spatial_shapes is not None else spatial_shapes)
            ir_merged = self.ir_gateway(target, self.ir_dropout(ir_target2))
            target = target + self.ir_alpha * (ir_merged - target)
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))
        return target

class Gate(nn.Module):

    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)

class Integral(nn.Module):
    """A static layer that calculates integral results from a distribution."""

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])

class LQE(nn.Module):
    """位置质量估计器 (Location Quality Estimator, LQE)"""

    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        """初始化 LQE 模块"""
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        """前向传播"""
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score

class TransformerDecoder(nn.Module):
    """Transformer 解码器，实现细粒度分布精炼 (Fine-grained Distribution Refinement, FDR)"""

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up, eval_idx=-1, layer_scale=2, act='relu'):
        """初始化 Transformer 解码器"""
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = (up, reg_scale, reg_max)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """预处理 MSDeformableAttention 的值（value）"""
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        """将模型转换为部署模式"""
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])

    def forward(self, target, ref_points_unact, memory, spatial_shapes, bbox_head, score_head, query_pos_head, pre_bbox_head, integral, up, reg_scale, attn_mask=None, memory_mask=None, dn_meta=None, ir_memory=None, ir_spatial_shapes=None):
        """前向传播"""
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
        ir_value = None
        if ir_memory is not None and ir_spatial_shapes is not None:
            ir_value = self.value_op(ir_memory, None, None, None, ir_spatial_shapes)
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                if ir_memory is not None and ir_spatial_shapes is not None:
                    ir_value = self.value_op(ir_memory, None, query_pos_embed.shape[-1], None, ir_spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed, ir_value=ir_value, ir_spatial_shapes=ir_spatial_shapes)
            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)
            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break
            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()
        return (torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores, output)

@register()
class DFINETransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, feat_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], num_levels=3, num_points=4, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0, activation='relu', num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, learn_query_content=False, eval_spatial_size=None, eval_idx=-1, eps=0.01, aux_loss=True, cross_attn_method='default', query_select_method='default', reg_max=32, reg_scale=4.0, layer_scale=1, mlp_act='relu', ir_feat_channels=None, ir_obj_boost=False, ir_obj_boost_norm='raw', ir_obj_alpha_mode='shared'):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        assert query_select_method in ('default', 'one2many', 'agnostic'), '查询选择方法无效'
        assert cross_attn_method in ('default', 'discrete'), '交叉注意力方法无效'
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self._build_input_proj_layer(feat_channels)
        self.ir_feat_channels = ir_feat_channels
        if ir_feat_channels is not None:
            self.ir_input_proj = nn.ModuleList()
            for in_ch in ir_feat_channels:
                if in_ch == hidden_dim:
                    self.ir_input_proj.append(nn.Identity())
                else:
                    self.ir_input_proj.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_ch, hidden_dim, 1, bias=False)), ('norm', nn.BatchNorm2d(hidden_dim))])))
        self.ir_obj_boost = ir_obj_boost and ir_feat_channels is not None
        self.ir_obj_boost_norm = ir_obj_boost_norm
        self.ir_obj_alpha_mode = ir_obj_alpha_mode
        if self.ir_obj_boost:
            self.ir_obj_heads = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_dim // 4), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim // 4, 1, 1)) for _ in ir_feat_channels])
            if ir_obj_alpha_mode == 'per_level':
                self.ir_obj_alpha = nn.Parameter(torch.zeros(len(ir_feat_channels)))
            else:
                self.ir_obj_alpha = nn.Parameter(torch.zeros(1))
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        _use_ir_aux = ir_feat_channels is not None
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method, use_ir_auxiliary=_use_ir_aux)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale, use_ir_auxiliary=_use_ir_aux)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead, reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)
        self.enc_output = nn.Sequential(OrderedDict([('proj', nn.Linear(hidden_dim, hidden_dim)), ('norm', nn.LayerNorm(hidden_dim))]))
        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)] + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)] + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList([self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))])

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)
        if self.ir_feat_channels is not None:
            for m, in_ch in zip(self.ir_input_proj, self.ir_feat_channels):
                if in_ch != self.hidden_dim:
                    init.xavier_uniform_(m[0].weight)
        if self.ir_obj_boost:
            for head in self.ir_obj_heads:
                for m in head.modules():
                    if isinstance(m, nn.Conv2d):
                        init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            init.constant_(m.bias, bias_init_with_prob(0.01))

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), ('norm', nn.BatchNorm2d(self.hidden_dim))])))
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)), ('norm', nn.BatchNorm2d(self.hidden_dim))])))
                in_channels = self.hidden_dim

    def _get_ir_objectness_boost(self, ir_feats: List[torch.Tensor], spatial_shapes):
        """从 IR 骨干特征生成 objectness heatmap, 并转换为 encoder token 维度的 boost 信号."""
        ir_obj_heatmaps = []
        boost_parts = []
        for lvl, feat in enumerate(ir_feats):
            proj_feat = self.ir_input_proj[lvl](feat)
            obj_logit = self.ir_obj_heads[lvl](proj_feat)
            ir_obj_heatmaps.append(obj_logit)
            if self.ir_obj_boost_norm == 'sigmoid':
                obj_boost = torch.sigmoid(obj_logit)
            elif self.ir_obj_boost_norm == 'tanh':
                obj_boost = torch.tanh(obj_logit)
            elif self.ir_obj_boost_norm == 'layer_zscore':
                mu = obj_logit.mean(dim=[2, 3], keepdim=True)
                std = obj_logit.std(dim=[2, 3], keepdim=True) + 1e-06
                obj_boost = (obj_logit - mu) / std
            else:
                obj_boost = obj_logit
            tgt_h, tgt_w = spatial_shapes[lvl]
            cur_h, cur_w = (obj_boost.shape[2], obj_boost.shape[3])
            if (cur_h, cur_w) != (tgt_h, tgt_w):
                obj_boost = F.interpolate(obj_boost, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
            boost_parts.append(obj_boost.flatten(2).permute(0, 2, 1))
        boost = torch.cat(boost_parts, dim=1)
        return (boost, ir_obj_heatmaps)

    def _extract_ir_obj_candidates(self, ir_obj_boost_signal, spatial_shapes, topk=50):
        """从 IR objectness boost 信号中提取 top-K 候选位置, 供 tracker 空间约束使用."""
        if self.ir_obj_boost_norm in ('sigmoid', 'tanh'):
            scores = ir_obj_boost_signal.squeeze(-1)
        else:
            scores = torch.sigmoid(ir_obj_boost_signal.squeeze(-1))
        device = scores.device
        positions = []
        for h, w in spatial_shapes:
            gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
            cx = (gx + 0.5) / w
            cy = (gy + 0.5) / h
            pos = torch.stack([cx, cy], dim=-1).reshape(-1, 2)
            positions.append(pos)
        positions = torch.cat(positions, dim=0)
        K = min(topk, scores.shape[1])
        topk_scores, topk_idx = torch.topk(scores, K, dim=-1)
        topk_pos = positions.unsqueeze(0).expand(scores.shape[0], -1, -1)
        topk_pos = topk_pos.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))
        candidates = torch.cat([topk_pos, topk_scores.unsqueeze(-1)], dim=-1)
        return candidates

    def _extract_ir_agreement_scores(self, ir_obj_heatmaps, pred_boxes):
        """为每个检测框采样其中心位置的 IR objectness 值 → per-detection agreement score."""
        B, N = pred_boxes.shape[:2]
        device = pred_boxes.device
        cx = pred_boxes[..., 0]
        cy = pred_boxes[..., 1]
        grid_x = cx * 2 - 1
        grid_y = cy * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(1)
        max_logits = torch.full((B, N), -1000000000.0, device=device)
        for heatmap in ir_obj_heatmaps:
            sampled = F.grid_sample(heatmap, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled = sampled.squeeze(1).squeeze(1)
            max_logits = torch.max(max_logits, sampled)
        return torch.sigmoid(max_logits)

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        feat_flatten = torch.concat(feat_flatten, 1)
        return (feat_flatten, spatial_shapes)

    def _get_ir_input(self, ir_feats: List[torch.Tensor]):
        """将IR骨干特征投影并展平，用于IR辅助memory"""
        proj_feats = [self.ir_input_proj[i](feat) for i, feat in enumerate(ir_feats)]
        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        feat_flatten = torch.concat(feat_flatten, 1)
        return (feat_flatten, spatial_shapes)

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * 2.0 ** lvl
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)
        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return (anchors, valid_mask)

    def _get_decoder_input(self, memory: torch.Tensor, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None, ir_obj_boost_signal=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory
        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)
        if ir_obj_boost_signal is not None:
            if self.ir_obj_alpha_mode == 'per_level':
                parts = ir_obj_boost_signal.split([h * w for h, w in spatial_shapes], dim=1)
                weighted_parts = [self.ir_obj_alpha[i] * parts[i] for i in range(len(parts))]
                ir_obj_boost_weighted = torch.cat(weighted_parts, dim=1)
            else:
                ir_obj_boost_weighted = self.ir_obj_alpha * ir_obj_boost_signal
            enc_outputs_logits = enc_outputs_logits + ir_obj_boost_weighted
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)
        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
        enc_topk_bboxes_list, enc_topk_logits_list = ([], [])
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)
        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        return (content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits)

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        topk_ind: torch.Tensor
        topk_anchors = outputs_anchors_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return (topk_memory, topk_logits, topk_anchors)

    def forward(self, feats, targets=None, ir_feats=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        ir_memory, ir_spatial_shapes = (None, None)
        if ir_feats is not None and self.ir_feat_channels is not None:
            ir_memory, ir_spatial_shapes = self._get_ir_input(ir_feats)
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(targets, self.num_classes, self.num_queries, self.denoising_class_embed, num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio, box_noise_scale=1.0)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (None, None, None, None)
        ir_obj_boost_signal, ir_obj_heatmaps = (None, None)
        if self.ir_obj_boost and ir_feats is not None:
            ir_obj_boost_signal, ir_obj_heatmaps = self._get_ir_objectness_boost(ir_feats, spatial_shapes)
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, ir_obj_boost_signal=ir_obj_boost_signal)
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits, query_features = self.decoder(init_ref_contents, init_ref_points_unact, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, self.pre_bbox_head, self.integral, self.up, self.reg_scale, attn_mask=attn_mask, dn_meta=dn_meta, ir_memory=ir_memory, ir_spatial_shapes=ir_spatial_shapes)
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1], 'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
            if self.ir_obj_boost and ir_obj_boost_signal is not None:
                out['ir_obj_candidates'] = self._extract_ir_obj_candidates(ir_obj_boost_signal, spatial_shapes)
                if ir_obj_heatmaps is not None:
                    out['ir_agreement_scores'] = self._extract_ir_agreement_scores(ir_obj_heatmaps, out_bboxes[-1])
            out['query_features'] = query_features
            corners = out_corners[-1]
            B_lc, N_lc = corners.shape[:2]
            prob = F.softmax(corners.reshape(B_lc, N_lc, 4, self.reg_max + 1), dim=-1)
            max_prob = prob.max(dim=-1).values
            out['loc_confidence'] = max_prob.mean(dim=-1)
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1], out_corners[-1], out_logits[-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta
            out['enc_outputs_logits'] = enc_outputs_logits
            if self.ir_obj_boost and ir_obj_heatmaps is not None and (targets is not None):
                ir_obj_loss = self._compute_ir_objectness_loss(ir_obj_heatmaps, targets, spatial_shapes)
                out['ir_obj_loss'] = ir_obj_loss
        return out

    def _compute_ir_objectness_loss(self, ir_obj_heatmaps, targets, spatial_shapes):
        """计算 IR objectness heatmap 的 BCE loss."""
        device = ir_obj_heatmaps[0].device
        batch_size = ir_obj_heatmaps[0].shape[0]
        total_loss = torch.tensor(0.0, device=device)
        n_levels = len(ir_obj_heatmaps)
        for lvl, obj_logit in enumerate(ir_obj_heatmaps):
            h, w = (obj_logit.shape[2], obj_logit.shape[3])
            gt_heatmap = torch.zeros(batch_size, 1, h, w, device=device)
            for b_idx in range(batch_size):
                if b_idx >= len(targets):
                    continue
                boxes = targets[b_idx].get('boxes', None)
                if boxes is None or boxes.shape[0] == 0:
                    continue
                for box in boxes:
                    cx, cy, bw, bh = box.tolist()
                    cx_f, cy_f = (cx * w, cy * h)
                    bw_f, bh_f = (bw * w, bh * h)
                    sigma_x = max(bw_f / 4, 1.0)
                    sigma_y = max(bh_f / 4, 1.0)
                    x_min = max(0, int(cx_f - 3 * sigma_x))
                    x_max = min(w, int(cx_f + 3 * sigma_x) + 1)
                    y_min = max(0, int(cy_f - 3 * sigma_y))
                    y_max = min(h, int(cy_f + 3 * sigma_y) + 1)
                    if x_min >= x_max or y_min >= y_max:
                        continue
                    ys = torch.arange(y_min, y_max, device=device, dtype=torch.float32)
                    xs = torch.arange(x_min, x_max, device=device, dtype=torch.float32)
                    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
                    gaussian = torch.exp(-((xx - cx_f) ** 2 / (2 * sigma_x ** 2) + (yy - cy_f) ** 2 / (2 * sigma_y ** 2)))
                    gt_heatmap[b_idx, 0, y_min:y_max, x_min:x_max] = torch.max(gt_heatmap[b_idx, 0, y_min:y_max, x_min:x_max], gaussian)
            pred = obj_logit.sigmoid()
            alpha_f = 0.25
            gamma = 2.0
            bce = F.binary_cross_entropy_with_logits(obj_logit, gt_heatmap, reduction='none')
            p_t = pred * gt_heatmap + (1 - pred) * (1 - gt_heatmap)
            focal_weight = alpha_f * gt_heatmap + (1 - alpha_f) * (1 - gt_heatmap)
            focal_weight = focal_weight * (1 - p_t) ** gamma
            loss_lvl = (focal_weight * bce).mean()
            total_loss = total_loss + loss_lvl
        return total_loss / max(n_levels, 1)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref, teacher_corners=None, teacher_logits=None):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d, 'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits} for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]
