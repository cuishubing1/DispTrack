"""   
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)    
Copyright(c) 2023 lyuwenyu. All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import re
import copy
from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict
from ..logger_module import get_logger
logger = get_logger(__name__)
from collections import defaultdict

class YAMLConfig(BaseConfig):

    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()
        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)
        self.yaml_cfg = copy.deepcopy(cfg)
        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model

    @property
    def postprocessor(self) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return super().criterion

    @property
    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided."""
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) and (not ('total_batch_size' in cfg and 'batch_size' in cfg)), '`batch_size` or `total_batch_size` should be choosed one'
        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from ..misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, 'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            _ = global_cfg[name].pop('total_batch_size')
        logger.info(f'building {name} with batch_size={bs}...')
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)
        return loader
