import os
import os.path as osp
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
import cv2


class FaceData(Dataset):
    def __init__(self, img_root, img_size=112) -> None:
        self.img_files = glob.glob(osp.join(img_root, "*", "*"), recursive=True)
        self.label_list = os.listdir(img_root)
        self.img_size = img_size
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
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]
        )

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = cv2.imread(img_file)
        img = self.transform(image=img)["image"]
        img = cv2.resize(img, (self.img_size, self.img_size))

        # img = img / 255.
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # img = torch.from_numpy(img).float()

        label_dir = Path(img_file).parent.name
        label = self.label_list.index(label_dir)
        return img, label

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    # img_files = glob.glob(osp.join("/dataset/dataset/face_test", "*", "*"), recursive=True)
    # print(len(img_files))
    # print(Path(img_files[0]))
    # print(Path(img_files[0]).parent.name)
    data = FaceData(img_root='/dataset/dataset/face_test')
    for d in data:
        print(d)
