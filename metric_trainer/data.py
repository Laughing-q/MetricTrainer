import os
import os.path as osp
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
import cv2


class FaceTrainData(Dataset):
    """For training"""
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


if __name__ == "__main__":
    # img_files = glob.glob(osp.join("/dataset/dataset/face_test", "*", "*"), recursive=True)
    # print(len(img_files))
    # print(Path(img_files[0]))
    # print(Path(img_files[0]).parent.name)
    data = FaceData(img_root='/dataset/dataset/face_test')
    for d in data:
        img, label = d
        print(label)
        cv2.imshow('p', img[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break
