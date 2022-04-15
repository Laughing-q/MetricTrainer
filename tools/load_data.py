import mxnet as mx
import os
import numpy as np
import cv2
import pickle
from torchvision import transforms
from mxnet import ndarray as nd


class Glint360Loader:
    def __init__(self, root_dir):
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        idx = self.imgidx[self.count]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        sample = mx.image.imdecode(img).asnumpy()
        self.count += 1
        return sample, label


def load_bin(path, image_size):
    with open(path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")  # py3
    data_list = []
    for flip in [0, 1]:
        data = nd.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
    print("test bin loaded done:", data_list[0].shape)
    return (data_list, issame_list)


if __name__ == "__main__":
    save_root = '/dataset/dataset/glint360k/'
    test =Glint360Loader(root_dir='/dataset/dataset/glint360k/glint360k')
    for img, label in test:
        print(label)
        cv2.imshow('p', img[:, :, ::-1])
        # cv2.imwrite('sample.jpg', img)
        if cv2.waitKey(0) == ord('q'):
            break
        print(label)

    # data_list, issame_list = load_bin(
    #     path="/dataset/dataset/glint360k/glint360k/lfw.bin", image_size=(112, 112)
    # )
    # one_pic = data_list[0][0]
    # print(one_pic)  # 3, 112, 112
    # print(type(one_pic))  # mxnet.ndarray
    # print(type(one_pic.asnumpy()))  # ndarray
    # one_pic = one_pic.asnumpy().astype(np.uint8)
    # one_pic = np.transpose(one_pic, (1, 2, 0))
    # print(one_pic.dtype)
    # cv2.imshow('p', one_pic)
    # cv2.waitKey(0)
