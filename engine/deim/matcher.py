"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved     
Modules to compute the matching cost and solve the corresponding LSAP.     

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, generalized_box_inner_iou
from ..core import register
import numpy as np
from ..logger_module import get_logger
logger = get_logger(__name__)

@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""
    __share__ = ['use_focal_loss']

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0, change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000):
        """Creates the matcher"""
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            logger.info(f'Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}')
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.epoch = -1
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0, num_queries_list=None):
        """Performs the matching"""
        bs, num_queries = outputs['pred_logits'].shape[:2]
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs['pred_logits'].flatten(0, 1))
        else:
            out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        if self.change_matcher and epoch >= self.matcher_change_epoch:
            class_score = out_prob[:, tgt_ids]
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = -1 * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * out_prob ** self.gamma * -(1 - out_prob + 1e-08).log()
                pos_cost_class = self.alpha * (1 - out_prob) ** self.gamma * -(out_prob + 1e-08).log()
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['boxes']) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        if num_queries_list:
            for i, qn in enumerate(num_queries_list):
                C[i, qn:] = 10000.0
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}
        return {'indices': indices}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_k])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1000000.0
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0), torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        return indices_list

    def set_epoch(self, epoch):
        self.epoch = epoch

    def identify_small_objects(self, boxes):
        """Identify small objects based on box area"""
        self.small_object_threshold = 0.0005
        areas = boxes[:, 2] * boxes[:, 3]
        is_small = areas < self.small_object_threshold
        return is_small
