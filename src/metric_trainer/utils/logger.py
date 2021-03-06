from loguru import logger
import os
import sys


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    # loguru_format = (
    #     "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    #     "<level>{level: <8}</level> | "
    #     "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    # )
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        # logger.add(save_file)
