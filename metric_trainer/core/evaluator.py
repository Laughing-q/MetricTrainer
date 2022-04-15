from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.pair import parse_pair
from ..data import FaceValData



class Evalautor:
    def __init__(self, pair_path, batch_size, img_size=112) -> None:
        imgl, imgr, flags, folds = parse_pair(pair_path=pair_path)
        self.flags = flags
        self.folds = folds
        self.dataset = FaceValData(imgl, imgr, img_size)
        self.val_loader = DataLoader(self.dataset, batch_size)

    def eval(self, model):
        model.cuda()
        pbar = tqdm(self.val_loader, total=len(self.val_loader))
        for imgl, imgr in pbar:
            imgl = imgl.cuda()
            imgl = imgl / 255.
            imgr = imgr.cuda()
            imgr = imgr / 255.

    def single_inference(self, img, model):
        img = img / 255.
