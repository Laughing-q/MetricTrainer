import torch.distributed as dist
import torch
import numpy as np
import random


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)


def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def is_main_process():

    return get_rank() == 0


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
