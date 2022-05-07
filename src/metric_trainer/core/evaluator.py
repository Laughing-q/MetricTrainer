from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import List
import torch
import sklearn
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from ..utils.pair import parse_pair
from ..utils.dist import get_rank
from ..data.dataset import FaceValData, ValBinData
from ..eval.verification import evaluate


# class Evalautor:
#     def __init__(self, pair_path, batch_size, img_size=112) -> None:
#         imgl, imgr, flags, folds = parse_pair(pair_path=pair_path)
#         self.flags = flags
#         self.folds = folds
#         self.dataset = FaceValData(imgl, imgr, img_size)
#         self.val_loader = DataLoader(self.dataset, batch_size)
#
#     def eval(self, model):
#         model.cuda()
#         pbar = tqdm(self.val_loader, total=len(self.val_loader))
#         for imgl, imgr in pbar:
#             imgl = imgl.cuda()
#             imgl = imgl / 255.
#             imgr = imgr.cuda()
#             imgr = imgr / 255.
#
#     def single_inference(self, img, model):
#         img = img / 255.


class Evalautor:
    """
    Eval image pairs from prepared *.bin file which download from insightface repo.
    """

    def __init__(
        self, val_targets, root_dir, image_size=112, batch_size=8, flip=True
    ) -> None:
        self.rank: int = get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.var_data_list: List[object] = []
        self.val_issame_list: List[object] = []
        self.var_name_list: List[str] = []
        self.flip = flip
        if self.rank == 0:
            self.init_dataset(
                val_targets=val_targets,
                data_dir=root_dir,
                image_size=image_size,
                batch_size=batch_size,
            )

    def init_dataset(self, val_targets, data_dir, image_size, batch_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                logger.info(f"loading val data {name}...")
                valdataset = ValBinData(path, image_size)
                valdataloader = DataLoader(
                    dataset=valdataset,
                    batch_size=batch_size,
                    # num_workers=nw,
                    shuffle=False,
                )
                self.var_data_list.append(valdataloader)
                self.val_issame_list.append(valdataset.issame_list)
                self.var_name_list.append(name)
                logger.info(f"load {len(valdataset) // 2} image pairs Done!")

    def val(self, model, nfolds=10, flip=True):
        model.cuda()
        accs, stds = [], []
        for i, val_data in enumerate(self.var_data_list):
            acc, std = self.val_one_data(
                val_data,
                self.val_issame_list[i],
                model,
                flip,
                nfolds,
            )
            accs.append(acc)
            stds.append(std)
            pf = "%20s" + "%20s" * 1
            print(pf % (self.var_name_list[i], '%1.5f+-%1.5f' % (acc, std)))
        return accs, stds

    def val_one_data(self, val_data, issame_list, model, flip, nfolds):
        s = ("%20s" + "%20s" * 1) % (
            f"Eval",
            f"Accuracy{'-Flip' if flip else ''}",
        )
        embeddings = []
        pbar = tqdm(enumerate(val_data), total=len(val_data))
        for i, imgs in pbar:
            imgs = imgs.cuda()
            imgs = imgs / 255.0
            if flip:
                imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)
            out = model(imgs)  # (N, 512)
            if flip:
                out = torch.split(out, out.shape[0] // 2, dim=0)  # ((N, 512), (N, 512))
                out = out[0] + out[1]  # (2, N, 512)
            # normalize
            out = normalize(out)
            embedding = out.detach().cpu().numpy()
            embeddings.append(embedding)
            pbar.set_description(s)
        embeddings = np.concatenate(embeddings, axis=0)
        _, _, accuracy, _, _, _ = evaluate(embeddings, issame_list, nrof_folds=nfolds)
        acc, std = np.mean(accuracy), np.std(accuracy)
        return acc, std
