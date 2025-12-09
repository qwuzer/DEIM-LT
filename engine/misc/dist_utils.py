"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import time
import random
import numpy as np
import atexit

import torch
import torch.nn as nn
import torch.distributed
import torch.backends.cudnn

from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data import DistributedSampler
# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader

# Global flag to track if distributed mode was successfully initialized
_dist_initialized_successfully = False


def setup_distributed(print_rank: int=0, print_method: str='builtin', seed: int=None, ):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    global _dist_initialized_successfully
    _dist_initialized_successfully = False
    
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv('RANK', -1))
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

        # For single GPU, skip distributed training (not needed)
        if WORLD_SIZE == 1:
            enabled_dist = False
            print('Single GPU detected, skipping distributed mode.')
            return enabled_dist

        # Determine backend: use nccl for CUDA, gloo for CPU
        if torch.cuda.is_available():
            backend = 'nccl'
            # Set environment variables to help NCCL in WSL2
            os.environ.setdefault('NCCL_DEBUG', 'WARN')
            os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
            os.environ.setdefault('NCCL_IB_DISABLE', '1')
            os.environ.setdefault('NCCL_P2P_DISABLE', '1')
        else:
            backend = 'gloo'
        
        # torch.distributed.init_process_group(backend=backend, init_method='env://')
        torch.distributed.init_process_group(backend=backend, init_method='env://')
        
        # Test if NCCL actually works by trying a simple operation
        if backend == 'nccl' and torch.cuda.is_available():
            try:
                # Test NCCL with a simple all_reduce
                test_tensor = torch.ones(1).cuda()
                torch.distributed.all_reduce(test_tensor)
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as test_e:
                # NCCL is broken, clean up and disable distributed
                if torch.distributed.is_initialized():
                    try:
                        torch.distributed.destroy_process_group()
                    except:
                        pass
                raise RuntimeError(f'NCCL test failed: {test_e}') from test_e
        
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
        enabled_dist = True
        _dist_initialized_successfully = True
        if get_rank() == print_rank:
            print('Initialized distributed mode...')

    except Exception as e:
        enabled_dist = False
        _dist_initialized_successfully = False
        # Clean up any partially initialized process group
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except:
                pass
        print(f'Not init distributed mode. Error: {e}')

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)

    return enabled_dist


def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process
    """
    import builtins as __builtin__

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    """Check if distributed mode is available and successfully initialized"""
    global _dist_initialized_successfully
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    # Only return True if we successfully initialized (not just partially)
    return _dist_initialized_successfully


@atexit.register
def cleanup():
    """cleanup distributed environment
    """
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)



def warp_model(
    model: torch.nn.Module,
    sync_bn: bool=False,
    dist_mode: str='ddp',
    find_unused_parameters: bool=False,
    compile: bool=False,
    compile_mode: str='reduce-overhead',
    **kwargs
):
    # Only use DDP if distributed mode is successfully initialized and world_size > 1
    if is_dist_available_and_initialized() and get_world_size() > 1:
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if dist_mode == 'dp':
            model = DP(model, device_ids=[rank], output_device=rank)
        elif dist_mode == 'ddp':
            try:
                model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
            except Exception as e:
                print(f'Warning: Failed to wrap model with DDP: {e}. Using model without DDP.')
        else:
            raise AttributeError('')
    else:
        # For single GPU or failed distributed init, just convert sync_bn if needed
        if sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model

def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers)
    return loader



def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time():
    """sync_time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()



def setup_seed(seed: int, deterministic=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


# for torch.compile
def check_compile():
    import torch
    import warnings
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok

def is_compile(model):
    import torch._dynamo
    return type(model) in (torch._dynamo.OptimizedModule, )

def de_complie(model):
    return model._orig_mod if is_compile(model) else model
