import os
import os.path as osp
import numbers
import torch
import pickle
import numpy as np
import glob
import mxnet as mx
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
import cv2
from torch.utils.data import distributed
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from metric_trainer.utils.dist import get_world_size

ROOT = "/data/datasets/face/glint360k/images"

def get_dataloader(dataset, is_dist, batch_size, workers):
    sampler = (
        distributed.DistributedSampler(dataset, shuffle=True)
        if is_dist
        else RandomSampler(dataset)
    )
    nw = min(
        [
            os.cpu_count() // get_world_size(),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=nw,
        drop_last=True,
    )
    return data_loader


class FaceTrainData(Dataset):
    """Read training data from folders"""

    def __init__(self, img_root, transform=None, img_size=112, rgb=True) -> None:
        # self.img_files, self.labels = self._get_imgs_and_labels_from_root(img_root)
        self.img_files, self.labels = self._get_imgs_and_labels_from_txt(img_root)
        self.img_size = img_size
        self.rgb = rgb
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    width=self.img_size,
                    height=self.img_size,
                    scale=(0.85, 1),
                    ratio=(1, 1),
                    p=transform.RandomResizedCrop,
                ),
                A.HorizontalFlip(p=transform.HorizontalFlip),
                # A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=transform.RandomBrightnessContrast),
            ]
        ) if transform is not None else None
        self.random_idx = np.arange(len(self))
        np.random.shuffle(self.random_idx)

    def _get_imgs_and_labels_from_root(self, img_root):
        """get images and labels from folers"""
        img_files = glob.glob(osp.join(img_root, "*", "*"), recursive=True)
        label_list = os.listdir(img_root)
        labels = [label_list.index(Path(im).parent.name) for im in img_files]
        return img_files, labels

    def _get_imgs_and_labels_from_txt(self, img_txt):
        """get images and labels a pre-written txt file"""
        with open(img_txt, 'r') as f:
            lines = f.read().strip().splitlines()
        img_files, labels = [], []
        for i, l in enumerate(lines):
            im, label = l.split(",")
            # img_files[i] = im.strip()
            # labels[i] = int(label.strip())
            img_files.append(im.strip())
            labels.append(int(label.strip()))
        return img_files, labels

    def __getitem__(self, index):
        # index = self.random_idx[index]
        img_file = self.img_files[index]
        img = cv2.imread(img_file)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = cv2.resize(img, (self.img_size, self.img_size))

        img = img.transpose(2, 0, 1)
        if self.rgb:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        # img = img / 255.

        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.img_files)


class FaceValData(Dataset):
    """For val

    Args:
        img_l (List[str]): list of images.
        img_r (List[str]): list of images.
        img_size (int): image size.
    """

    def __init__(self, img_l, img_r, img_size=112, rgb=True) -> None:
        self.img_l = img_l
        self.img_r = img_r
        self.img_size = img_size
        self.rgb = rgb

        assert len(self.img_l) == len(self.img_r)

    def __len__(self):
        return len(self.img_l)

    def __getitem__(self, index):
        imgl = self.get_one_img(self.img_l[index])
        imgr = self.get_one_img(self.img_r[index])
        return imgl, imgr

    def get_one_img(self, file):
        img = cv2.imread(file)
        img = cv2.resize(img, (self.img_size, self.img_size))
        if len(img.shape) == 2:  # 如果为灰度图
            img = np.stack([img] * 3, 2)  # 沿着2轴（通道处）连续stack三份，转为三通道
        img = img.transpose(2, 0, 1)
        if self.rgb:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)


class Glint360Data(Dataset):
    """Read training data from glint360k"""

    def __init__(self, root_dir, transform=None, img_size=112, rgb=False):
        self.img_size = img_size
        self.rgb = rgb
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    width=self.img_size,
                    height=self.img_size,
                    scale=(0.85, 1),
                    ratio=(1, 1),
                    p=transform.RandomResizedCrop,
                ),
                A.HorizontalFlip(p=transform.HorizontalFlip),
                # A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=transform.RandomBrightnessContrast),
            ]
        ) if transform is not None else None
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        # self.labels = np.load(labels)
        # save labels
        # self.labels = []
        # for idx in tqdm(self.imgidx, total=len(self.imgidx)):
        #     self.labels.append(self._get_label(self.imgrec.read_idx(idx)))
        # np.save("labels.npy", np.array(self.labels))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        # NOTE: the original image from train.rec of glint360k is `RGB` format
        img = mx.image.imdecode(img).asnumpy()

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = img.transpose(2, 0, 1)
        if self.rgb:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        return img, int(label)

    def _get_label(self, s):
        header, _ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        return label

    def __len__(self):
        return len(self.imgidx)


class ValBinData(Dataset):
    """
    Read image pairs from prepared *.bin file which download from insightface repo.

    Args:
        bin_file: bin file prepared from insightface.
        img_size: image size.
        rgb: swap `bgr` channels to `rgb` if True.
    """

    def __init__(self, bin_file, img_size=112, rgb=False) -> None:
        super().__init__()
        try:
            with open(bin_file, "rb") as f:
                self.bins, self.issame_list = pickle.load(f)  # py2
        except UnicodeDecodeError:
            with open(bin_file, "rb") as f:
                self.bins, self.issame_list = pickle.load(f, encoding="bytes")  # py3
        self.img_size = img_size
        self.rgb = rgb

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, index):
        img = mx.image.imdecode(self.bins[index])
        if img.shape[1] != self.img_size:
            img = mx.image.resize_short(img, self.img_size)
        # NOTE: the original image from *.bin files in glint360k is `RGB` format
        img = img.asnumpy()
        if self.rgb:
            img = img[..., ::-1]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        return img


if __name__ == "__main__":
    from tqdm import tqdm
    # data = FaceTrainData(img_root='./images.txt')
    data = Glint360Data(root_dir="/data/datasets/face/glint360k")
    dataloader = get_dataloader(data, False, batch_size=128, workers=4)
    for i, d in enumerate(dataloader):
        img, label = d
        print(i, label.shape)
    # tt = 0
    # for i, d in enumerate(data):
    #     img, label, t, n = d
    #     tt += t
    #     # print(t)
    #     if i % 128 == 0:
    #         print(i, tt, n)
    #         tt = 0
        # img2 = Image.fromarray(img)
    #     img2.show()
    #     print(img.shape)
    #     print(label)
    #     cv2.imshow("p", img)
    #     if cv2.waitKey(0) == ord("q"):
    #         break

    # data = ValBinData(bin_file="/d/dataset/face/glint360k/lfw.bin")
    # for img in data:
    #     img2 = Image.fromarray(img)
    #     img2.show()
    #     print(img.shape)
    #     cv2.imshow("p", img)
    #     if cv2.waitKey(0) == ord("q"):
    #         break
