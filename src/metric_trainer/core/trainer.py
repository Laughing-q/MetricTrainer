from timm import create_model
from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
import os.path as osp
import os
from pytorch_metric_learning import losses, reducers, samplers
from ..data.dataset import FaceTrainData, Glint360Loader, get_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from ..models.partial_fc import PartialFC
from ..utils.lr_scheduler import PolyScheduler
from ..utils.dist import get_world_size, get_rank
from ..utils.callbacks import CallBackVerification, CallBackLogging
from ..utils.metric import AverageMeter
from ..models.losses import CombinedMarginLoss
from ..utils.logger import setup_logger


def build_metric(name, embedding_dim, num_class, sample_rate, fp16):
    reducer = reducers.MeanReducer()
    if name == "arcface":
        loss_func = losses.ArcFaceLoss(
            num_classes=num_class, embedding_size=embedding_dim, reducer=reducer
        )
    elif name == "circleloss":
        loss_func = losses.CircleLoss(m=0.25, gamma=256, reducer=reducer)
    elif name == "tripletloss":
        loss_func = losses.TripletMarginLoss(reducer=reducer)
    elif name == "partial_fc":
        margin_loss = CombinedMarginLoss(
            64,
            1.0,
            0.0,
            0.4,
        )
        loss_func = PartialFC(
            margin_loss,
            embedding_dim,
            num_class,
            sample_rate,
            fp16,
        )
    return loss_func


def build_dataset(data, *args, **kwargs):
    if data == "glint360k":
        return Glint360Loader(*args, **kwargs)
    elif data == "folder":
        return FaceTrainData(*args, **kwargs)


class Trainer:
    def __init__(self, cfg, local_rank) -> None:
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = local_rank

        self.cfg = cfg
        self.backbone = cfg.MODEL.BACKBONE
        self.embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.num_classes = cfg.MODEL.NUM_CLASS
        # batch size for one gpu
        self.batch_size = cfg.SOLVER.BATCH_SIZE_PER_GPU
        self.fp16 = cfg.SOLVER.FP16
        self.max_epoch = cfg.SOLVER.NUM_EPOCH
        self.save_dir = cfg.OUTPUT

        self.dataset = build_dataset(
            cfg.DATASET.TYPE, cfg.DATASET.TRAIN, cfg.DATASET.IMG_SIZE
        )

        # class-aware args
        self.sample_rate = cfg.MODEL.SAMPLE_RATE

    def before_train(self):
        model = create_model(
            model_name=self.backbone,
            num_classes=self.embedding_dim,
            pretrained=True,
            global_pool="avg",
        )
        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                bucket_cap_mb=16,
                find_unused_parameters=True,
            )
            self.model._set_static_graph()
        self.model = model
        self.model.train().cuda()

        self.train_loader = get_dataloader(
            self.dataset, self.is_distributed, self.batch_size, self.cfg.NUM_WORKERS
        )
        self.max_iter = len(self.train_loader)

        self.loss_func = build_metric(
            self.cfg.MODEL.LOSS,
            self.embedding_dim,
            self.num_classes,
            self.sample_rate,
            self.fp16,
        )

        self.loss_func.train().cuda()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

        self.optimizer = optim.SGD(
            params=[
                {"params": self.model.parameters()},
                {"params": self.loss_func.parameters()},
            ],
            lr=self.cfg.SOLVER.BASE_LR,
            momentum=self.cfg.SOLVER.MOMENTUM,
            weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
        )
        total_batch_size = self.cfg.SOLVER.BATCH_SIZE_PER_GPU * get_world_size()
        warmup_step = (
            self.cfg.DATASET.NUM_IMAGES
            // total_batch_size
            * self.cfg.SOLVER.WARMUP_EPOCH
        )
        total_step = (
            self.cfg.DATASET.NUM_IMAGES // total_batch_size * self.cfg.SOLVER.NUM_EPOCH
        )
        self.lr_scheduler = PolyScheduler(
            optimizer=self.optimizer,
            base_lr=self.cfg.SOLVER.BASE_LR,
            max_steps=total_step,
            warmup_steps=warmup_step,
        )

        self.callback_verification = CallBackVerification(
            val_targets=self.cfg.DATASET.VAL_TARGETS, rec_prefix=self.cfg.DATASET.VAL
        )
        self.logging = CallBackLogging(
            self.cfg.SOLVER.LOGGER_STEP,
            total_step,
            self.batch_size,
        )
        self.loss_am = AverageMeter()

        os.makedirs(self.save_dir, exist_ok=True)
        setup_logger(
            self.save_dir,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        self.train_in_epoch()

    def train_in_epoch(self):
        for self.epoch in range(self.max_epoch):
            self.train_in_iter()

    def train_in_iter(self):
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        for self.iter, (imgs, labels) in enumerate(self.train_loader):
            imgs = imgs.cuda()
            imgs = imgs / 255.0
            labels = labels.cuda()

            embeddings = self.model(imgs)
            loss = self.loss_func(embeddings, labels, self.optimizer)
            self.optimizer.zero_grad()
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

            self.lr_scheduler.step()
            with torch.no_grad():
                self.loss_am.update(loss.item())
                self.logging(
                    # self.global_iter(),
                    self.iter,
                    self.max_iter,
                    self.loss_am,
                    self.epoch,
                    self.max_epoch,
                    self.fp16,
                    self.lr_scheduler.get_last_lr()[0],
                    self.scaler,
                )
                if (
                    self.global_iter() % self.cfg.SOLVER.VAL_STEP == 0
                    and self.global_iter() > 50
                ):
                    self.callback_verification(self.global_iter(), self.model)
                    self.save_ckpt()
        # with torch.no_grad():
        #     self.callback_verification(self.global_iter(), self.model)
        # self.save_ckpt()

    def global_iter(self):
        return self.iter + self.max_iter * self.epoch

    def save_ckpt(self):
        path_pfc = osp.join(self.cfg.OUTPUT, "softmax_fc_gpu_{}.pt".format(self.rank))
        torch.save(self.loss_func.state_dict(), path_pfc)
        model = (
            self.model.module
            if type(self.model)
            in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
            else self.model
        )
        ckpt = {
            "epoch": self.epoch,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.rank == 0:
            path_module = osp.join(self.cfg.OUTPUT, "model.pt")
            torch.save(ckpt, path_module)

    def eval(self):
        pass
