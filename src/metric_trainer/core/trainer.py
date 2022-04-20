from timm import create_model
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from pytorch_metric_learning import losses, reducers, samplers
from ..data.dataset import FaceTrainData, Glint360Loader


def build_metric(name, embedding_dim, num_class):
    reducer = reducers.MeanReducer()
    if name == "arcface":
        loss_func = losses.ArcFaceLoss(
            num_classes=num_class, embedding_size=embedding_dim, reducer=reducer
        )
    elif name == "circleloss":
        loss_func = losses.CircleLoss(m=0.25, gamma=256, reducer=reducer)
    elif name == "tripletloss":
        loss_func = losses.TripletMarginLoss(reducer=reducer)
    return loss_func

def build_dataset(data, *args, **kwargs):
    if data == 'glint360k':
        return Glint360Loader(*args, **kwargs)
    elif data == 'folder':
        return FaceTrainData(*args, **kwargs)


class Trainer:
    def __init__(self, cfg) -> None:
        self.model = create_model(
            model_name=cfg.MODEL.BACKBONE,
            num_classes=cfg.MODEL.EMBEDDING_DIM,
            pretrained=False,
            global_pool="avg",
        )
        self.loss_func = build_metric(
            cfg.MODEL.LOSS, cfg.MODEL.EMBEDDING_DIM, cfg.MODEL.NUM_CLASS
        )

        # TODO
        self.dataset = build_dataset(
            cfg.DATASET.TYPE, cfg.DATASET.TRAIN, 'labels.npy', cfg.DATASET.IMG_SIZE
        )
        # sampler = samplers.MPerClassSampler(self.dataset.labels, m=4, batch_size=cfg.SOLVER.BATCH_SIZE)
        self.train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            # sampler=sampler,
            # shuffle=True,
        )
        self.optimizer = optim.SGD(
            [
                {"params": self.model.parameters()},
                {"params": self.loss_func.parameters()},
            ],
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    def train(self):
        self.model.cuda()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for _, (img, label) in pbar:
            img = img.cuda()
            img = img / 255.0
            label = label.cuda()

            self.optimizer.zero_grad()
            embeddings = self.model(img)
            loss = self.loss_func(embeddings, label)
            pbar.desc = f"loss: {loss}"

            loss.backward()
            self.optimizer.step()
