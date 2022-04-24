import os
import time
from typing import List
from loguru import logger
import torch


class CallBackSaveLog(object):
    def __init__(self, save_dir, val_targets) -> None:
        val_targets = ["val/" + val for val in val_targets]
        self.save_dir = save_dir
        self.keys = [
            "epoch",
            "lr",
            "img_size",
        ] + val_targets

    def __call__(self, vals):
        if not os.path.exists(self.save_dir):
            print(f"{self.save_dir} is not existed, skip!")
            return 

        file = os.path.join(self.save_dir, "results.txt")
        n = len(vals)  # number of cols
        s = (
            ""
            if os.path.exists(file)
            else (("%20s," * n % tuple(self.keys)).rstrip(",") + "\n")
        )  # add header
        with open(file, "a") as f:
            f.write(s + ("%20.5g," * n % tuple(vals)).rstrip(",") + "\n")
