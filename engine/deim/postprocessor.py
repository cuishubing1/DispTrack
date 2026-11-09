"""    
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from ..core import register
__all__ = ['PostProcessor', 'SegPostProcessor', 'DQPostProcessor']

def mod(a, b):
    out = a - a // b * b
    return out

@register()
class PostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        """额外的表示方法，在打印模型结构时显示"""
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False):
        """前向传播函数，处理检测头的输出，将其转换为最终检测结果。"""
        logits, boxes = (outputs['pred_logits'], outputs['pred_boxes'])
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        if self.deploy_mode:
            return (labels, boxes, scores)
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()]).to(boxes.device).reshape(labels.shape)
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        return results

    def deploy(self):
        """设置模型为部署模式（推理模式）"""
        self.eval()
        self.deploy_mode = True
        return self

@register()
class SegPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category', 'eval_spatial_size', 'eval_mask_ratio']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, eval_spatial_size=None, eval_mask_ratio=1) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.eval_spatial_size = eval_spatial_size
        self.eval_mask_ratio = eval_mask_ratio
        self.deploy_mode = False

    def extra_repr(self) -> str:
        """额外的表示方法，在打印模型结构时显示"""
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False):
        """前向传播函数，处理检测头的输出，将其转换为最终检测结果。"""
        logits, boxes, pred_masks = (outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_masks'])
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            pred_masks = pred_masks.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
                masks = pred_masks.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1]))
        masks = []
        for i in range(pred_masks.size(0)):
            masks_i = self.process_mask(pred_masks[i], boxes[i], orig_target_sizes[i].tolist(), for_eval)
            masks.append(masks_i)
        if self.deploy_mode:
            return (labels, boxes, scores, masks)
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()]).to(boxes.device).reshape(labels.shape)
        results = []
        for i, (lab, box, sco, mask) in enumerate(zip(labels, boxes, scores, masks)):
            result = dict(labels=lab, boxes=box, scores=sco, masks=mask)
            results.append(result)
        return results

    def deploy(self):
        """设置模型为部署模式（推理模式）"""
        self.eval()
        self.deploy_mode = True
        return self

    def process_mask(self, pred_masks, boxes, orig_target_sizes, for_eval=False):
        """处理掩码：插值到原始图像尺寸并将边界框外的区域置为0"""
        w, h = orig_target_sizes
        if for_eval:
            boxes = boxes / torch.from_numpy(np.array([w / (self.eval_spatial_size[1] / self.eval_mask_ratio), h / (self.eval_spatial_size[0] / self.eval_mask_ratio)])).unsqueeze(0).repeat(1, 2).to(boxes.device)
            w = int(self.eval_spatial_size[1] / self.eval_mask_ratio)
            h = int(self.eval_spatial_size[0] / self.eval_mask_ratio)
        H_mask, W_mask = pred_masks.shape[-2:]
        if H_mask != int(h) or W_mask != int(w):
            masks_i = F.interpolate(pred_masks.unsqueeze(1), size=(int(h), int(w)), mode='bilinear', align_corners=False).squeeze(1)
        else:
            masks_i = pred_masks
        masks_i = masks_i > 0.0
        if not self.deploy_mode:
            num_queries = masks_i.shape[0]
            for i in range(num_queries):
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(int(w), x2)
                y2 = min(int(h), y2)
                mask_filtered = torch.zeros_like(masks_i[i])
                mask_filtered[y1:y2, x1:x2] = masks_i[i, y1:y2, x1:x2]
                masks_i[i] = mask_filtered
        return masks_i

@register()
class DQPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        """额外的表示方法，在打印模型结构时显示"""
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False):
        """前向传播函数，处理检测头的输出，将其转换为最终检测结果。"""
        logits, boxes = (outputs['pred_logits'], outputs['pred_boxes'])
        if 'num_queries_list' in outputs:
            num_queries_list = outputs['num_queries_list']
            B, N, C = logits.shape
            device = logits.device
            mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            for i, qn in enumerate(num_queries_list):
                mask[i, :qn] = True
            mask = mask.unsqueeze(-1).expand(B, N, C)
            if self.use_focal_loss:
                logits = torch.where(mask, logits, torch.tensor(-10000.0, device=device))
            else:
                logits = torch.where(mask, logits, torch.tensor(0.0, device=device))
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), max(num_queries_list), dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > max(num_queries_list):
                scores, index = torch.topk(scores, max(num_queries_list), dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        if self.deploy_mode:
            return (labels, boxes, scores)
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()]).to(boxes.device).reshape(labels.shape)
        results = []
        for lab, box, sco, qn in zip(labels, boxes, scores, num_queries_list):
            result = dict(labels=lab[:qn], boxes=box[:qn], scores=sco[:qn])
            results.append(result)
        return results

    def deploy(self):
        """设置模型为部署模式（推理模式）"""
        self.eval()
        self.deploy_mode = True
        return self
