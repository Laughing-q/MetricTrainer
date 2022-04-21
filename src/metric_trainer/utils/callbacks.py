import os
import time
from typing import List
from loguru import logger

import torch

from ..eval.verification import test, load_bin
from ..utils.metric import AverageMeter
from torch import distributed


class CallBackVerification(object):
    def __init__(
        self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112)
    ):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank == 0:
            self.init_dataset(
                val_targets=val_targets, data_dir=rec_prefix, image_size=image_size
            )

        self.summary_writer = summary_writer

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = test(
                self.ver_list[i], backbone, 10, 10
            )
            logger.info(
                "[%s][%d]XNorm: %f" % (self.ver_name_list[i], global_step, xnorm)
            )
            logger.info(
                "[%s][%d]Accuracy-Flip: %1.5f+-%1.5f"
                % (self.ver_name_list[i], global_step, acc2, std2)
            )

            # self.summary_writer: SummaryWriter
            # self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logger.info(
                "[%s][%d]Accuracy-Highest: %1.5f"
                % (self.ver_name_list[i], global_step, self.highest_acc_list[i])
            )
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank == 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size

        self.init = False
        self.tic = 0

    def __call__(
        self,
        current_iter: int,
        max_iter: int,
        loss: AverageMeter,
        epoch: int,
        fp16: bool,
        learning_rate: float,
        grad_scaler: torch.cuda.amp.GradScaler,
    ):
        if self.rank == 0 and current_iter > 0 and current_iter % self.frequent == 0:
            if self.init:
                try:
                    speed: float = (
                        self.frequent * self.batch_size / (time.time() - self.tic)
                    )
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float("inf")

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((current_iter + 1) / self.total_step)
                time_for_end = time_total - time_now
                if fp16:
                    msg = (
                        "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Step: %d/%d   "
                        "Fp16 Grad Scale: %2.f   Required: %1.f hours"
                        % (
                            speed_total,
                            loss.avg,
                            learning_rate,
                            epoch,
                            current_iter,
                            max_iter,
                            grad_scaler.get_scale(),
                            time_for_end,
                        )
                    )
                else:
                    msg = (
                        "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Step: %d/%d   "
                        "Required: %1.f hours"
                        % (
                            speed_total,
                            loss.avg,
                            learning_rate,
                            epoch,
                            current_iter,
                            max_iter,
                            time_for_end,
                        )
                    )
                logger.info(msg)
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
