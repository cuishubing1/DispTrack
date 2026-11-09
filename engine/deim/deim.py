"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""
import torch
import torch.nn as nn
from ..core import register
__all__ = ['DEIM', 'DualModalDEIM', 'DualDecoderDEIM']

@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

@register()
class DualModalDEIM(nn.Module):
    __inject__ = ['backbone_rgb', 'backbone_ir', 'fusion', 'encoder', 'decoder']

    def __init__(self, backbone_rgb: nn.Module, backbone_ir: nn.Module, fusion: nn.Module, encoder: nn.Module, decoder: nn.Module, **kwargs):
        super().__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_ir = backbone_ir
        self.fusion = fusion
        self.decoder = decoder
        self.encoder = encoder

    def _build_track_quality_prior(self, targets, device):
        if targets is None:
            return (None, None)
        quality_list = []
        has_prev_list = []
        for t in targets:
            has_prev = float(t.get('has_prev_frame', torch.tensor([0.0], device=device)).reshape(-1)[0].item())
            has_prev_list.append(has_prev)
            if has_prev < 0.5:
                quality_list.append(0.5)
                continue
            cur_boxes = t.get('boxes', None)
            prev_boxes = t.get('prev_boxes', None)
            if cur_boxes is None or prev_boxes is None:
                quality_list.append(0.5)
                continue
            n_cur = float(cur_boxes.shape[0])
            n_prev = float(prev_boxes.shape[0])
            if n_cur < 1 or n_prev < 1:
                quality_list.append(0.0)
                continue
            count_consistency = min(n_cur, n_prev) / max(n_cur, n_prev)
            quality_list.append(count_consistency)
        track_quality_prior = torch.tensor(quality_list, dtype=torch.float32, device=device)
        has_prev_mask = torch.tensor(has_prev_list, dtype=torch.float32, device=device)
        return (track_quality_prior, has_prev_mask)

    def forward(self, x_rgb, x_ir, targets=None, prev_rgb=None, prev_ir=None, temporal_meta=None):
        feat_rgb = self.backbone_rgb(x_rgb)
        feat_ir = self.backbone_ir(x_ir)
        prev_feat_rgb, prev_feat_ir = (None, None)
        if prev_rgb is not None and prev_ir is not None:
            prev_feat_rgb = self.backbone_rgb(prev_rgb)
            prev_feat_ir = self.backbone_ir(prev_ir)
        track_quality_prior, has_prev_mask = self._build_track_quality_prior(targets, x_rgb.device)
        if temporal_meta is not None and isinstance(temporal_meta, dict):
            if 'has_prev_frame' in temporal_meta:
                has_prev_mask = temporal_meta['has_prev_frame'].to(x_rgb.device).float()
        if targets is not None and hasattr(self.fusion, 'set_tracking'):
            self.fusion.set_tracking(True)
        try:
            feat_fused = self.fusion(feat_rgb, feat_ir, prev_feat_rgb=prev_feat_rgb, prev_feat_ir=prev_feat_ir, track_quality_prior=track_quality_prior, has_prev_mask=has_prev_mask)
        except TypeError:
            feat_fused = self.fusion(feat_rgb, feat_ir)
        align_gain = None
        if targets is not None and hasattr(self.fusion, 'compute_alignment_gain'):
            try:
                align_gain = self.fusion.compute_alignment_gain(targets)
            except Exception as e:
                align_gain = 0.0
            self.fusion.set_tracking(False)
        x = self.encoder(feat_fused)
        x = self.decoder(x, targets, ir_feats=feat_ir)
        if align_gain is not None:
            if isinstance(x, dict):
                x['align_gain'] = torch.tensor(align_gain).to(x_rgb.device)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

@register()
class DualDecoderDEIM(nn.Module):
    """双 Decoder 双模态检测模型 (v12)."""
    __inject__ = ['backbone_rgb', 'backbone_ir', 'fusion', 'encoder', 'encoder_ir', 'decoder_rgb', 'decoder_ir']

    def __init__(self, backbone_rgb: nn.Module, backbone_ir: nn.Module, fusion: nn.Module, encoder: nn.Module, encoder_ir: nn.Module, decoder_rgb: nn.Module, decoder_ir: nn.Module, **kwargs):
        super().__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_ir = backbone_ir
        self.fusion = fusion
        self.encoder = encoder
        self.encoder_ir = encoder_ir
        self.decoder_rgb = decoder_rgb
        self.decoder_ir = decoder_ir

    def _build_track_quality_prior(self, targets, device):
        if targets is None:
            return (None, None)
        quality_list = []
        has_prev_list = []
        for t in targets:
            has_prev = float(t.get('has_prev_frame', torch.tensor([0.0], device=device)).reshape(-1)[0].item())
            has_prev_list.append(has_prev)
            if has_prev < 0.5:
                quality_list.append(0.5)
                continue
            cur_boxes = t.get('boxes', None)
            prev_boxes = t.get('prev_boxes', None)
            if cur_boxes is None or prev_boxes is None:
                quality_list.append(0.5)
                continue
            n_cur = float(cur_boxes.shape[0])
            n_prev = float(prev_boxes.shape[0])
            if n_cur < 1 or n_prev < 1:
                quality_list.append(0.0)
                continue
            count_consistency = min(n_cur, n_prev) / max(n_cur, n_prev)
            quality_list.append(count_consistency)
        track_quality_prior = torch.tensor(quality_list, dtype=torch.float32, device=device)
        has_prev_mask = torch.tensor(has_prev_list, dtype=torch.float32, device=device)
        return (track_quality_prior, has_prev_mask)

    def forward(self, x_rgb, x_ir, targets=None, prev_rgb=None, prev_ir=None, temporal_meta=None):
        feat_rgb = self.backbone_rgb(x_rgb)
        feat_ir = self.backbone_ir(x_ir)
        prev_feat_rgb, prev_feat_ir = (None, None)
        if prev_rgb is not None and prev_ir is not None:
            prev_feat_rgb = self.backbone_rgb(prev_rgb)
            prev_feat_ir = self.backbone_ir(prev_ir)
        track_quality_prior, has_prev_mask = self._build_track_quality_prior(targets, x_rgb.device)
        if temporal_meta is not None and isinstance(temporal_meta, dict):
            if 'has_prev_frame' in temporal_meta:
                has_prev_mask = temporal_meta['has_prev_frame'].to(x_rgb.device).float()
        if targets is not None and hasattr(self.fusion, 'set_tracking'):
            self.fusion.set_tracking(True)
        try:
            feat_fused = self.fusion(feat_rgb, feat_ir, prev_feat_rgb=prev_feat_rgb, prev_feat_ir=prev_feat_ir, track_quality_prior=track_quality_prior, has_prev_mask=has_prev_mask)
        except TypeError:
            feat_fused = self.fusion(feat_rgb, feat_ir)
        align_gain = None
        if targets is not None and hasattr(self.fusion, 'compute_alignment_gain'):
            try:
                align_gain = self.fusion.compute_alignment_gain(targets)
            except Exception:
                align_gain = 0.0
            self.fusion.set_tracking(False)
        fused_memory = self.encoder(feat_fused)
        ir_memory = self.encoder_ir(feat_ir)
        out_rgb = self.decoder_rgb(fused_memory, targets)
        out_ir = self.decoder_ir(ir_memory, targets)
        if align_gain is not None:
            if isinstance(out_rgb, dict):
                out_rgb['align_gain'] = torch.tensor(align_gain).to(x_rgb.device)
        if self.training:
            out = out_rgb
            out['ir_pred_logits'] = out_ir['pred_logits']
            out['ir_pred_boxes'] = out_ir['pred_boxes']
            if out_ir.get('pred_corners') is not None:
                out['ir_pred_corners'] = out_ir['pred_corners']
            for key in ('aux_outputs', 'enc_aux_outputs', 'pre_outputs', 'dn_outputs', 'dn_pre_outputs', 'dn_meta', 'enc_outputs_logits'):
                if key in out_ir:
                    out[f'ir_{key}'] = out_ir[key]
        else:
            out = out_rgb
            out['ir_pred_logits'] = out_ir['pred_logits']
            out['ir_pred_boxes'] = out_ir['pred_boxes']
            if 'query_features' in out_ir:
                out['ir_query_features'] = out_ir['query_features']
        return out

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
