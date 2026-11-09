"""
DEIM: DETR with Improved Matching for Fast Convergence    
Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
---------------------------------------------------------------------------------   
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate
import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode
import random
from functools import partial
from collections import defaultdict
from ..core import register
torchvision.disable_beta_transforms_warning()
from copy import deepcopy
from PIL import Image, ImageDraw
import os
from ..logger_module import get_logger
logger = get_logger(__name__)
__all__ = ['DataLoader', 'BaseCollateFunction', 'BatchImageCollateFunction', 'DualModalBatchImageCollateFunction', 'TemporalDualModalBatchImageCollateFunction', 'batch_image_collate_fn']

@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += '\n'
            format_string += '    {0}: {1}'.format(n, getattr(self, n))
        format_string += '\n)'
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle

@register()
def batch_image_collate_fn(items):
    """only batch image"""
    return (torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items])

class BaseCollateFunction(object):

    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')

def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales

@register()
class BatchImageCollateFunction(BaseCollateFunction):

    def __init__(self, stop_epoch=None, ema_restart_decay=0.9999, base_size=640, base_size_repeat=None, mixup_prob=0.0, mixup_epochs=[0, 0], copyblend_prob=0.0, copyblend_epochs=[0, 0], copyblend_type='blend', conflict_with_mixup=False, area_threshold=100, num_objects=3, with_expand=False, expand_ratios=[0.1, 0.25], random_num_objects=False, data_vis=False, vis_save='./vis_dataset/') -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        self.mixup_prob, self.mixup_epochs = (mixup_prob, mixup_epochs)
        self.copyblend_prob, self.copyblend_epochs, self.copyblend_type = (copyblend_prob, copyblend_epochs, copyblend_type)
        self.area_threshold, self.num_objects = (area_threshold, num_objects)
        self.data_vis, self.vis_save = (data_vis, vis_save)
        self.with_expand, self.expand_ratios, self.random_num_objects = (with_expand, expand_ratios, random_num_objects)
        self.conflict_with_mixup = conflict_with_mixup
        if self.mixup_prob > 0 or self.copyblend_prob > 0:
            self.data_vis, self.vis_save = (data_vis, vis_save)
            os.makedirs(self.vis_save, exist_ok=True) if self.data_vis else None
            if self.mixup_prob > 0:
                logger.info('     ### Using MixUp with Prob@{} in {} epochs ### '.format(self.mixup_prob, self.mixup_epochs))
            if self.copyblend_prob > 0:
                logger.info('     ### Using CopyBlend-{} with Prob@{} in {} epochs ### '.format(copyblend_type, copyblend_prob, copyblend_epochs))
                logger.info(f'     ### CopyBlend -- area threshold@{area_threshold} and num of object@{num_objects} ###     ')
                if self.with_expand:
                    logger.info(f'     ### CopyBlend -- expand@{expand_ratios} ###     ')
                if self.random_num_objects:
                    logger.info(f'     ### CopyBlend -- random num of objects@{[1, self.num_objects]} ###     ')
        if stop_epoch is not None:
            logger.info('     ### Multi-scale Training until {} epochs ### '.format(self.stop_epoch))
            logger.info('     ### Multi-scales@ {} ###        '.format(self.scales))
        self.print_info_flag = True
        self.print_copyblend_flag = True

    def apply_mixup(self, images, targets):
        """Applies Mixup augmentation to the batch if conditions are met."""
        if self.epoch == self.mixup_epochs[-1] and self.print_info_flag:
            logger.info(f'     ### Attention --- Mixup is closed after epoch@ {self.epoch} ###')
            self.print_info_flag = False
        MixUp_flag, CopyBlend_flag = (False, False)
        beta = round(random.uniform(0.45, 0.55), 6)
        if random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]:
            beta = round(random.uniform(0.45, 0.55), 6)
            MixUp_flag = True
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)
            for i in range(len(targets)):
                updated_targets[i]['boxes'] = torch.cat([targets[i]['boxes'], shifted_targets[i]['boxes']], dim=0)
                updated_targets[i]['labels'] = torch.cat([targets[i]['labels'], shifted_targets[i]['labels']], dim=0)
                updated_targets[i]['area'] = torch.cat([targets[i]['area'], shifted_targets[i]['area']], dim=0)
                if 'masks' in targets[i]:
                    updated_targets[i]['masks'] = torch.cat([targets[i]['masks'], shifted_targets[i]['masks']], dim=0)
                updated_targets[i]['mixup'] = torch.tensor([beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']), dtype=torch.float32)
            targets = updated_targets
        elif self.copyblend_epochs[0] <= self.epoch < self.copyblend_epochs[-1] and random.random() < self.copyblend_prob:
            if self.epoch == self.copyblend_epochs[-1] and self.print_copyblend_flag:
                print(f'     ### Attention --- CopyBlend closed after epoch@ {self.epoch} ###')
                self.print_copyblend_flag = False
            CopyBlend_flag = True
            objects_pool = defaultdict(list)
            img_height, img_width = images[0].shape[-2:]
            for i in range(len(images)):
                source_boxes = targets[i]['boxes']
                source_labels = targets[i]['labels']
                source_areas = targets[i]['area']
                source_masks = targets[i].get('masks', None)
                valid_objects = [idx for idx in range(len(source_boxes)) if source_areas[idx] >= self.area_threshold]
                for idx in valid_objects:
                    objects_pool['boxes'].append(source_boxes[idx])
                    objects_pool['labels'].append(source_labels[idx])
                    objects_pool['areas'].append(source_areas[idx])
                    objects_pool['image_idx'].append(i)
                    objects_pool['image_height'].append(img_height)
                    objects_pool['image_width'].append(img_width)
                    if source_masks is not None:
                        objects_pool['masks'].append(source_masks[idx])
            if len(objects_pool['boxes']) == 0:
                return (images, targets)
            for key in ['boxes', 'labels', 'areas', 'masks']:
                if key != 'masks':
                    objects_pool[key] = torch.stack(objects_pool[key]) if objects_pool[key] else torch.tensor([])
                elif key == 'masks':
                    objects_pool[key] = torch.stack(objects_pool[key]) if key in objects_pool and objects_pool[key] else None
            batch_size = len(images)
            updated_images = images.clone()
            updated_targets = deepcopy(targets)
            for i in range(batch_size):
                if self.random_num_objects:
                    num_objects = random.randint(1, min(self.num_objects, len(objects_pool['boxes'])))
                else:
                    num_objects = min(self.num_objects, len(objects_pool['boxes']))
                selected_indices = random.sample(range(len(objects_pool['boxes'])), num_objects)
                blend_boxes = []
                blend_labels = []
                blend_areas = []
                blend_mixup_ratios = []
                blend_masks = []
                for idx in selected_indices:
                    box = objects_pool['boxes'][idx]
                    label = objects_pool['labels'][idx]
                    area = objects_pool['areas'][idx]
                    source_idx = objects_pool['image_idx'][idx]
                    source_height = objects_pool['image_height'][idx]
                    source_width = objects_pool['image_width'][idx]
                    source_mask = objects_pool['masks'][idx] if objects_pool['masks'] is not None else None
                    cx, cy, w, h = box
                    x1_src, y1_src = (int((cx - w / 2) * source_width), int((cy - h / 2) * source_height))
                    x2_src, y2_src = (int((cx + w / 2) * source_width), int((cy + h / 2) * source_height))
                    x1_src, y1_src = (max(x1_src, 0), max(y1_src, 0))
                    x2_src, y2_src = (min(x2_src, img_width), min(y2_src, img_height))
                    new_w_px, new_h_px = (x2_src - x1_src, y2_src - y1_src)
                    if new_w_px <= 0 or new_h_px <= 0:
                        continue
                    x1 = random.randint(0, img_width - new_w_px) if new_w_px < img_width else 0
                    y1 = random.randint(0, img_height - new_h_px) if new_h_px < img_height else 0
                    x2, y2 = (x1 + new_w_px, y1 + new_h_px)
                    new_cx, new_cy = ((x1 + new_w_px / 2) / img_width, (y1 + new_h_px / 2) / img_height)
                    new_w, new_h = (new_w_px / img_width, new_h_px / img_height)
                    blend_boxes.append(torch.tensor([new_cx, new_cy, new_w, new_h]))
                    blend_labels.append(label)
                    blend_areas.append(area)
                    blend_mixup_ratios.append(1.0 - beta)
                    if source_mask is not None:
                        new_mask = torch.zeros((img_height, img_width), dtype=source_mask.dtype, device=source_mask.device)
                        cropped_mask = source_mask[y1_src:y2_src, x1_src:x2_src]
                        new_mask[y1:y2, x1:x2] = cropped_mask
                        blend_masks.append(new_mask)
                    if self.with_expand:
                        alpha = round(random.uniform(self.expand_ratios[0], self.expand_ratios[1]), 6)
                        expand_w, expand_h = (int(new_w_px * alpha), int(new_h_px * alpha))
                        x1_expand, y1_expand = (x1_src - max(x1_src - expand_w, 0), y1_src - max(y1_src - expand_h, 0))
                        x2_expand, y2_expand = (min(x2_src + expand_w, img_width) - x2_src, min(y2_src + expand_h, img_height) - y2_src)
                        new_x1_expand, new_y1_expand = (x1 - max(x1 - x1_expand, 0), y1 - max(y1 - y1_expand, 0))
                        new_x2_expand, new_y2_expand = (min(x2 + x2_expand, img_width) - x2, min(y2 + y2_expand, img_height) - y2)
                        x1_src, y1_src, x2_src, y2_src = (x1_src - new_x1_expand, y1_src - new_y1_expand, x2_src + new_x2_expand, y2_src + new_y2_expand)
                        x1, y1, x2, y2 = (x1 - new_x1_expand, y1 - new_y1_expand, x2 + new_x2_expand, y2 + new_y2_expand)
                        if source_mask is not None:
                            new_mask = torch.zeros((img_height, img_width), dtype=source_mask.dtype, device=source_mask.device)
                            cropped_mask = source_mask[y1_src:y2_src, x1_src:x2_src]
                            new_mask[y1:y2, x1:x2] = cropped_mask
                            blend_masks[-1] = new_mask
                    copy_patch_orig = images[source_idx, :, y1_src:y2_src, x1_src:x2_src]
                    if self.copyblend_type == 'blend':
                        if source_mask is not None:
                            mask_region = source_mask[y1_src:y2_src, x1_src:x2_src].unsqueeze(0)
                            blended_patch = updated_images[i, :, y1:y2, x1:x2] * beta + copy_patch_orig * (1 - beta)
                            updated_images[i, :, y1:y2, x1:x2] = updated_images[i, :, y1:y2, x1:x2] * (1 - mask_region) + blended_patch * mask_region
                        else:
                            blended_patch = updated_images[i, :, y1:y2, x1:x2] * beta + copy_patch_orig * (1 - beta)
                            updated_images[i, :, y1:y2, x1:x2] = blended_patch
                    elif source_mask is not None:
                        mask_region = source_mask[y1_src:y2_src, x1_src:x2_src].unsqueeze(0)
                        updated_images[i, :, y1:y2, x1:x2] = updated_images[i, :, y1:y2, x1:x2] * (1 - mask_region) + copy_patch_orig * mask_region
                    else:
                        updated_images[i, :, y1:y2, x1:x2] = copy_patch_orig
                if len(blend_boxes) > 0:
                    blend_boxes = torch.stack(blend_boxes)
                    blend_labels = torch.stack(blend_labels)
                    blend_areas = torch.stack(blend_areas)
                    updated_targets[i]['mixup'] = torch.tensor([1.0] * len(updated_targets[i]['boxes']) + blend_mixup_ratios, dtype=torch.float32)
                    updated_targets[i]['boxes'] = torch.cat([updated_targets[i]['boxes'], blend_boxes])
                    updated_targets[i]['labels'] = torch.cat([updated_targets[i]['labels'], blend_labels])
                    updated_targets[i]['area'] = torch.cat([updated_targets[i]['area'], blend_areas])
                    if len(blend_masks) > 0 and 'masks' in updated_targets[i]:
                        blend_masks = torch.stack(blend_masks)
                        updated_targets[i]['masks'] = torch.cat([updated_targets[i]['masks'], blend_masks])
            images, targets = (updated_images, updated_targets)
        return (images, targets)

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        images, targets = self.apply_mixup(images, targets)
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'].unsqueeze(1), size=sz, mode='nearest').squeeze(1)
        return (images, targets)

@register()
class DualModalBatchImageCollateFunction(BatchImageCollateFunction):

    def __call__(self, items):
        images_rgb = torch.cat([x[0][0][None] for x in items], dim=0)
        images_ir = torch.cat([x[0][1][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images_rgb = F.interpolate(images_rgb, size=sz)
            images_ir = F.interpolate(images_ir, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'].unsqueeze(1), size=sz, mode='nearest').squeeze(1)
        return ((images_rgb, images_ir), targets)

@register()
class TemporalDualModalBatchImageCollateFunction(BatchImageCollateFunction):

    def __call__(self, items):
        images_rgb_cur = torch.cat([x[0]['current'][0][None] for x in items], dim=0)
        images_ir_cur = torch.cat([x[0]['current'][1][None] for x in items], dim=0)
        images_rgb_prev = torch.cat([x[0]['previous'][0][None] for x in items], dim=0)
        images_ir_prev = torch.cat([x[0]['previous'][1][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images_rgb_cur = F.interpolate(images_rgb_cur, size=sz)
            images_ir_cur = F.interpolate(images_ir_cur, size=sz)
            images_rgb_prev = F.interpolate(images_rgb_prev, size=sz)
            images_ir_prev = F.interpolate(images_ir_prev, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'].unsqueeze(1), size=sz, mode='nearest').squeeze(1)
        temporal_meta = {'is_first_frame': torch.cat([tg['is_first_frame'] for tg in targets], dim=0), 'has_prev_frame': torch.cat([tg['has_prev_frame'] for tg in targets], dim=0), 'prev_index': torch.cat([tg['prev_index'] for tg in targets], dim=0), 'sequence_index': torch.cat([tg['sequence_index'] for tg in targets], dim=0), 'frame_index_in_sequence': torch.cat([tg['frame_index_in_sequence'] for tg in targets], dim=0)}
        return (((images_rgb_cur, images_ir_cur), (images_rgb_prev, images_ir_prev), temporal_meta), targets)
