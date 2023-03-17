from contextlib import contextmanager
import torch.distributed as dist
import torch


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


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])
