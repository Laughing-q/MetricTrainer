import torch
from metric_trainer.core.evaluator import Evalautor
from timm import create_model
from lqcv.utils.timer import Timer
from omegaconf import OmegaConf
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/partial_glint360k.yaml",
        help="config file",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="",
        help="val data path, if the argument is empty, then eval the val data from config",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default="",
        help="weight path",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    cfg = OmegaConf.load(opt.config)

    model = create_model(
        model_name=cfg.MODEL.BACKBONE,
        num_classes=cfg.MODEL.EMBEDDING_DIM,
        pretrained=False,
        global_pool="avg",
    )
    ckpt = torch.load(opt.weight)
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    # img = torch.rand((1, 3, 112, 112), dtype=torch.float32).cuda()
    # time = Timer(start=True, round=2, unit="ms")
    # for i in tqdm(range(100), total=100):
    #     model(img)
    # print(f"average inference time: {time.since_start() / 100}ms")
    # exit()
    testor = Evalautor(
        val_targets=["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw", "vgg2_fp"],
        root_dir=opt.data,
        batch_size=opt.batch_size,
    )
    with torch.no_grad():
        testor.val(model, flip=True)
