# modified from torchvision to also return the union 
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""
import numpy as np
import torch
from torch import Tensor
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w.clamp(min=0.0), y_c - 0.5 * h.clamp(min=0.0), x_c + 0.5 * w.clamp(min=0.0), y_c + 0.5 * h.clamp(min=0.0)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.stack(b, dim=-1)

def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return (iou, union)

def shrink_boxes(boxes, ratio=1.5):
    """按比例缩小边界框（保持中心点不变）"""
    assert boxes.shape[-1] == 4, f'Expected last dimension to be 4, but got shape {boxes.shape}'
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    cy = (boxes[..., 1] + boxes[..., 3]) / 2
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    new_w = w * ratio
    new_h = h * ratio
    if isinstance(boxes, torch.Tensor):
        shrunk_boxes = torch.empty_like(boxes)
    else:
        shrunk_boxes = np.empty_like(boxes)
    shrunk_boxes[..., 0] = cx - new_w / 2
    shrunk_boxes[..., 1] = cy - new_h / 2
    shrunk_boxes[..., 2] = cx + new_w / 2
    shrunk_boxes[..., 3] = cy + new_h / 2
    return shrunk_boxes

def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/"""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area

def generalized_box_inner_iou(boxes1, boxes2, ratio=2.0):
    """Generalized IoU from https://giou.stanford.edu/"""
    boxes2 = shrink_boxes(boxes2, ratio)
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks"""
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~masks.bool(), 100000000.0).flatten(1).min(-1)[0]
    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~masks.bool(), 100000000.0).flatten(1).min(-1)[0]
    return torch.stack([x_min, y_min, x_max, y_max], 1)
