# MetricTrainer
- This framework is for metric study, built on [timm](https://github.com/rwightman/pytorch-image-models) and [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning).
- This repo is experimental.

## TODO
- [X] logger
- [X] eval
- [X] DDP
- [X] amp training
- [X] dataloader
- [X] saving model stuff
- [X] resume
- [X] plot results
- [ ] support other optimizer
- [ ] support other scheduler
- [ ] ema
- [ ] inference


## Quick Start

<details open>
<summary>Installation</summary>

Clone repo and install [requirements.txt](https://github.com/Laughing-q/yolov5-q/blob/master/requirements.txt) in a
**Python>=3.7.0** environment, including**PyTorch>=1.7.1**.

```shell
pip install pytorch-metric-learning
pip install git+https://github.com/Laughing-q/lqcv.git
pip install git+https://github.com/Laughing-q/pytorch-image-models.git
git clone https://github.com/Laughing-q/MetricTrainer.git
cd MetricTrainer
pip install -r requirements.txt
pip install -e .
```

</details>

<details open>
<summary>Training</summary>

- Prepare your own config, see `configs/` for more details.
- `partial fc` rely on `DDP` mode, so if you get only one GPU, just set `n=1`.
- Multi GPU(DDP)
```shell
python -m torch.distributed.run --nproc_per_node=n tools/train.py -c configs/test.yaml
```
</details>

<details open>
<summary>Eval</summary>

- prepare your val datasets like below(you can get the datasets from [insightface](https://github.com/deepinsight/insightface.git)):
```plain
├── root_dir
│   ├── lfw.bin
│   ├── cfp_fp.bin
│   ├── agedb_30.bin
│   ├── calfw.bin
│   ├── cplfw.bin
│   ├── vgg2_fp.bin
```

```shell
python tools/eval.py -c configs/partial_glint360k.yaml -d root_dir \
    -w runs/CosFace_noaug/best.pt
```

</details>

## Tips
- I haven't test resume at all, maybe there will get some bug.
