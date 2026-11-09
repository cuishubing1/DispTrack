"""Process helpers for the core training and inference modules."""
import torch

def is_dist_available_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()

def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()

def Multiprocess_sync():
    if is_dist_available_and_initialized():
        torch.distributed.barrier()

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model
