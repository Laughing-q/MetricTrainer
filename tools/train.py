from metric_trainer.core import Trainer
from omegaconf import OmegaConf
import os
import torch
import torch.distributed as dist
from loguru import logger

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=WORLD_SIZE,
            rank=RANK)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(LOCAL_RANK)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


cfg = OmegaConf.load('configs/partial_glint360k.yaml')

if LOCAL_RANK != -1:
    init_distributed()

trainer = Trainer(cfg, LOCAL_RANK)
trainer.train()

if WORLD_SIZE > 1 and RANK == 0:
    logger.info("Destroying process group... ")
    dist.destroy_process_group()
