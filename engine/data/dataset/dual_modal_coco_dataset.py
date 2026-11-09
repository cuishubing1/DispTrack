"""     
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os, cv2, contextlib
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register
from typing import Any, Callable, Optional, Tuple, Union
torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
__all__ = ['DualModalCocoDetection']

def _frame_key(name, infrared=False):
    parts = name.replace('\\', '/').split('/')
    if 'img1' not in parts:
        raise ValueError('Expected SEQUENCE/img1/FRAME in COCO file_name')
    index = parts.index('img1')
    sequence = parts[index - 1]
    if infrared:
        if not sequence.endswith('T'):
            raise ValueError('IR sequence must end in T')
        sequence = sequence[:-1]
    return (sequence, parts[-1])

class DualModalCocoDetection_Cache(torchvision.datasets.CocoDetection):

    def __init__(self, root_rgb: Union[str, Path], root_ir: Union[str, Path], annFile: str, annFile_ir: Optional[str]=None, transform: Optional[Callable[..., Any]]=None, target_transform: Optional[Callable[..., Any]]=None, transforms: Optional[Callable[..., Any]]=None, cache_imgsz: Optional[int]=None, ram_cache: bool=False, ir_channels: int=3) -> None:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(root_rgb, annFile, transform, target_transform, transforms)
        if ram_cache:
            raise ValueError('RAM caching is not part of the core release')
        self.root_ir = root_ir
        self.cache_imgsz = cache_imgsz
        self.ram_cache = ram_cache
        self.ir_channels = ir_channels
        self.cache_flag = False
        self.ni = len(self.ids)
        self.ims, self.labs = ({i: None for i in self.ids}, {i: None for i in self.ids})
        self.coco_ir = None
        if annFile_ir:
            from pycocotools.coco import COCO
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.coco_ir = COCO(annFile_ir)
        if self.coco_ir is None:
            raise ValueError('Paired IR annotations are required')
        ir_by_frame = {}
        for ir_id, image in self.coco_ir.imgs.items():
            key = _frame_key(image['file_name'], infrared=True)
            if key in ir_by_frame:
                raise ValueError(f'Duplicate IR frame: {key}')
            ir_by_frame[key] = ir_id
        self.ir_image_ids = {}
        seen_rgb = set()
        for rgb_id in self.ids:
            key = _frame_key(self.coco.imgs[rgb_id]['file_name'])
            if key in seen_rgb:
                raise ValueError(f'Duplicate RGB frame: {key}')
            seen_rgb.add(key)
            if key not in ir_by_frame:
                raise ValueError(f'RGB frame has no paired IR annotation: {key}')
            self.ir_image_ids[rgb_id] = ir_by_frame[key]
        extras = set(ir_by_frame) - seen_rgb
        if extras:
            raise ValueError(f'IR annotations have unpaired frames: {sorted(extras)[:5]}')

    def _load_target(self, id: int) -> Any:
        rgb_anns = super()._load_target(id)
        if hasattr(self, 'coco_ir') and self.coco_ir is not None:
            ir_anns = self.coco_ir.loadAnns(self.coco_ir.getAnnIds(imgIds=self.ir_image_ids[id]))
            return {'rgb': rgb_anns, 'ir': ir_anns}
        return rgb_anns
        return (i, self.ims[i], self.labs[i])

    def _load_image(self, id):
        rgb_name = self.coco.loadImgs(id)[0]['file_name']
        ir_name = self.coco_ir.loadImgs(self.ir_image_ids[id])[0]['file_name']

        def relative_name(name):
            parts = name.replace('\\', '/').split('/')
            if 'img1' not in parts:
                raise ValueError('Expected SEQUENCE/img1/FRAME in COCO file_name')
            i = parts.index('img1')
            return Path(*parts[i - 1:])
        rgb_path, ir_path = (relative_name(rgb_name), relative_name(ir_name))
        if rgb_path.parts[0] + 'T' != ir_path.parts[0] or rgb_path.name != ir_path.name:
            raise ValueError(f'Unpaired COCO image ID {id}: {rgb_name} / {ir_name}')
        return (Image.open(Path(self.root) / rgb_path).convert('RGB'), Image.open(Path(self.root_ir) / ir_path).convert('L' if self.ir_channels == 1 else 'RGB'))

    def __getitem__(self, index):
        image_id = self.ids[index]
        return (self._load_image(image_id), self._load_target(image_id))

@register()
class DualModalCocoDetection(DualModalCocoDetection_Cache, DetDataset):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category', 'cache_imgsz', 'ram_cache']

    def __init__(self, img_folder_rgb, img_folder_ir, ann_file, transforms, return_masks=False, remap_mscoco_category=False, cache_imgsz=None, ram_cache=False, ir_channels=3, ann_file_ir=None, **kwargs):
        super(DualModalCocoDetection, self).__init__(img_folder_rgb, img_folder_ir, ann_file, annFile_ir=ann_file_ir, cache_imgsz=cache_imgsz, ram_cache=ram_cache, ir_channels=ir_channels)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder_rgb = img_folder_rgb
        self.img_folder_ir = img_folder_ir
        self.ann_file = ann_file
        self.ann_file_ir = ann_file_ir
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return (img, target)

    def load_item(self, idx):
        image, target = super(DualModalCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target_dict = {'image_id': image_id}
        if isinstance(target, dict) and 'rgb' in target and ('ir' in target):
            target_dict['annotations'] = target['rgb']
            target_dict['ir_annotations'] = target['ir']
        else:
            target_dict['annotations'] = target
        if self.remap_mscoco_category:
            image, target_dict = self.prepare(image, target_dict, category2label=mscoco_category2label)
        else:
            image, target_dict = self.prepare(image, target_dict)
        target_dict['idx'] = torch.tensor([idx])
        if 'boxes' in target_dict:
            target_dict['boxes'] = convert_to_tv_tensor(target_dict['boxes'], key='boxes', spatial_size=image[0].size[::-1])
        if 'ir_boxes' in target_dict:
            target_dict['ir_boxes'] = convert_to_tv_tensor(target_dict['ir_boxes'], key='boxes', spatial_size=image[0].size[::-1])
        if 'masks' in target_dict:
            target_dict['masks'] = convert_to_tv_tensor(target_dict['masks'], key='masks')
        return (image, target_dict)

    def extra_repr(self) -> str:
        s = f' img_folder_rgb: {self.img_folder_rgb}\n img_folder_ir: {self.img_folder_ir}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self):
        return self.coco.dataset['categories']

    @property
    def category2name(self):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    valid_mask_flags = []
    for polygons in segmentations:
        if not polygons:
            valid_mask_flags.append(False)
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        if isinstance(polygons, list):
            is_valid = False
            for poly in polygons:
                if isinstance(poly, list) and len(poly) >= 6:
                    is_valid = True
                    break
            if not is_valid:
                valid_mask_flags.append(False)
                masks.append(torch.zeros((height, width), dtype=torch.uint8))
                continue
        try:
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in polygons:
                pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            if mask.sum() == 0:
                valid_mask_flags.append(False)
            else:
                valid_mask_flags.append(True)
            masks.append(mask)
        except Exception as e:
            valid_mask_flags.append(False)
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return (masks, torch.tensor(valid_mask_flags, dtype=torch.bool))

class ConvertCocoPolysToMask(object):

    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Union[Image.Image, Tuple[Image.Image, Image.Image]], target, **kwargs):
        if isinstance(image, tuple):
            w, h = image[0].size
        else:
            w, h = image.size
        image_id = target['image_id']
        image_id = torch.tensor([image_id])
        anno = target['annotations']
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        ir_boxes = None
        if 'ir_annotations' in target:
            ir_anno = target['ir_annotations']
            ir_anno = [obj for obj in ir_anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
            ir_map = {}
            for obj in ir_anno:
                if 'instance_id' in obj:
                    ir_map[obj['instance_id']] = obj['bbox']
                elif 'track_id' in obj:
                    ir_map[obj['track_id']] = obj['bbox']
                elif 'id' in obj and 'instance_id' not in obj:
                    ir_map[obj['id']] = obj['bbox']
            matched_ir_boxes = []
            valid_indices = []
            if ir_map:
                for idx, obj in enumerate(anno):
                    iid = obj.get('instance_id')
                    if iid is None:
                        iid = obj.get('track_id')
                    if iid is None and 'id' in obj and ('instance_id' not in obj):
                        iid = obj.get('id')
                    if iid is not None and iid in ir_map:
                        matched_ir_boxes.append(ir_map[iid])
                        valid_indices.append(idx)
                if len(matched_ir_boxes) > 0:
                    if len(matched_ir_boxes) == len(anno):
                        ir_boxes = torch.as_tensor(matched_ir_boxes, dtype=torch.float32).reshape(-1, 4)
                        ir_boxes[:, 2:] += ir_boxes[:, :2]
                        ir_boxes[:, 0::2].clamp_(min=0, max=w)
                        ir_boxes[:, 1::2].clamp_(min=0, max=h)
                    else:
                        ir_boxes = None
                else:
                    ir_boxes = None
                    pass
            if not ir_map or (ir_map and len(matched_ir_boxes) == 0):
                if len(anno) == len(ir_anno) and len(anno) > 0:
                    ir_boxes = [obj['bbox'] for obj in ir_anno]
                    ir_boxes = torch.as_tensor(ir_boxes, dtype=torch.float32).reshape(-1, 4)
                    ir_boxes[:, 2:] += ir_boxes[:, :2]
                    ir_boxes[:, 0::2].clamp_(min=0, max=w)
                    ir_boxes[:, 1::2].clamp_(min=0, max=h)
                else:
                    ir_boxes = None
        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj['category_id']] for obj in anno]
        else:
            labels = [obj['category_id'] for obj in anno]
        labels = torch.tensor(labels, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj['segmentation'] for obj in anno]
            masks, valid_mask_flags = convert_coco_poly_to_mask(segmentations, h, w)
        keypoints = None
        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if self.return_masks:
            keep = keep & valid_mask_flags
        boxes = boxes[keep]
        if ir_boxes is not None:
            ir_boxes = ir_boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        target = {}
        target['boxes'] = boxes
        if ir_boxes is not None:
            target['ir_boxes'] = ir_boxes
        target['labels'] = labels
        if self.return_masks:
            target['masks'] = masks
        target['image_id'] = image_id
        if keypoints is not None:
            target['keypoints'] = keypoints
        area = torch.tensor([obj['area'] for obj in anno])
        iscrowd = torch.tensor([obj['iscrowd'] if 'iscrowd' in obj else 0 for obj in anno])
        target['area'] = area[keep]
        target['iscrowd'] = iscrowd[keep]
        target['orig_size'] = torch.as_tensor([int(w), int(h)])
        return (image, target)
mscoco_category2name = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
