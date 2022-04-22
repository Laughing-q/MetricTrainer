import torch
import os
from typing import List
from loguru import logger
from metric_trainer.eval.verification import test, load_bin
from timm import create_model


class CallBackVerification(object):
    def __init__(self, val_targets, rec_prefix, image_size=(112, 112)):
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.init_dataset(
            val_targets=val_targets, data_dir=rec_prefix, image_size=image_size
        )

    def ver_test(self, backbone: torch.nn.Module):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = test(
                self.ver_list[i], backbone, 10, 10
            )
            logger.info("[%s]XNorm: %f" % (self.ver_name_list[i], xnorm))
            logger.info(
                "[%s]Accuracy-Flip: %1.5f+-%1.5f" % (self.ver_name_list[i], acc2, std2)
            )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logger.info(
                "[%s]Accuracy-Highest: %1.5f"
                % (self.ver_name_list[i], self.highest_acc_list[i])
            )
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, backbone: torch.nn.Module):
        backbone.eval()
        self.ver_test(backbone)


if __name__ == "__main__":
    model = create_model(
        model_name='cspresnet50',
        num_classes=512,
        pretrained=False,
        global_pool="avg",
    )
    ckpt = torch.load('runs/model.pt')
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    callback_verification = CallBackVerification(
        val_targets=["lfw", "cfp_fp", "agedb_30"],
        rec_prefix="/dataset/dataset/glint360k/glint360k",
    )
    callback_verification(model)
    # LFW: 0.99683
    # cfp_fp: 0.96514
    # agedb_30: 0.97200
