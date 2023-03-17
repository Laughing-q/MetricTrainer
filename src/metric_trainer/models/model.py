import timm
from functools import partial
from torch import nn
from .common import get_norm, get_activation


def build_model(cfg):
    act_layer = get_activation(name=cfg.ACT_LAYER)
    norm_layer = get_norm(name=cfg.NORM_LAYER)

    Backbone = partial(
        timm.create_model,
        model_name=cfg.BACKBONE,
        num_classes=0,
        global_pool="avg" if cfg.POOLING else "",
        pretrained=cfg.PRETRAINED,
        # norm_layer=norm_layer,  # cspnet does not support norm_layer and act_layer
        # act_layer=act_layer,
        # head_norm_first=True,
        exportable=True,
    )
    if "convnext" in cfg.BACKBONE and (
        "nano" not in cfg.BACKBONE and "tiny" not in cfg.BACKBONE
    ):
        backbone = Backbone(conv_mlp=True)
    else:
        backbone = Backbone()

    return FaceModel(
        backbone=backbone,
        num_features=cfg.EMBEDDING_DIM,
        drop_ratio=cfg.get("DROP_RATIO", 0.0),
        pool=cfg.POOLING,
    )


class FaceModel(nn.Module):
    def __init__(self, backbone, num_features=512, drop_ratio=0.0, pool=False) -> None:
        super().__init__()
        self.fc_scale = 1 if pool else 4 * 4
        self.channels = backbone.num_features
        self.num_features = num_features

        # output N, C, 4, 4 from (112, 112)
        self.backbone = backbone
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(self.channels),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(self.channels * self.fc_scale, self.num_features),
            # nn.BatchNorm1d(self.num_features),
        )
        self.features = nn.BatchNorm1d(self.num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 2:
            b, c = x.shape
            x = x.view(b, c, 1, 1)
        x = self.output_layer(x)
        x = self.features(x)
        return x
