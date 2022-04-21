import os
import os.path as osp
import numbers
import torch
import numpy as np
import glob
import mxnet as mx
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
import cv2
from tqdm import tqdm
from torch.utils.data import distributed
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader


def get_dataloader(dataset, is_dist, batch_size, workers):
    sampler = (
        distributed.DistributedSampler(dataset, shuffle=True)
        if is_dist
        else RandomSampler(dataset)
    )
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=nw
    )
    return data_loader


class FaceTrainData(Dataset):
    """Read training data from folders"""

    def __init__(self, img_root, img_size=112, rgb=True) -> None:
        self.img_files = glob.glob(osp.join(img_root, "*", "*"), recursive=True)
        self.label_list = os.listdir(img_root)
        self.img_size = img_size
        self.rgb = rgb
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    width=self.img_size,
                    height=self.img_size,
                    scale=(0.85, 1),
                    ratio=(1, 1),
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]
        )
        self.labels = [
            self.label_list.index(Path(im).parent.name) for im in self.img_files
        ]

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = cv2.imread(img_file)
        img = self.transform(image=img)["image"]
        img = cv2.resize(img, (self.img_size, self.img_size))

        img = img.transpose(2, 0, 1)
        if self.rgb:
            img = img[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        # img = img / 255.

        label_dir = Path(img_file).parent.name
        label = self.label_list.index(label_dir)
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


class Glint360Loader(Dataset):
    """Read training data from glint360k"""

    def __init__(self, root_dir, img_size=112, rgb=True):
        self.img_size = img_size
        self.rgb = rgb
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    width=self.img_size,
                    height=self.img_size,
                    scale=(0.85, 1),
                    ratio=(1, 1),
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]
        )
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
        sample = mx.image.imdecode(img).asnumpy()
        img = self.transform(image=sample)["image"]

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


if __name__ == "__main__":
    # data = FaceTrainData(img_root='/dataset/dataset/face_test')
    data = Glint360Loader(root_dir="/dataset/dataset/glint360k/glint360k")
    for d in data:
        img, label = d
        print(img.shape)
        print(label)
        cv2.imshow("p", img[:, :, ::-1])
        if cv2.waitKey(0) == ord("q"):
            break
