#/bin/bash
python -m torch.distributed.run --nproc_per_node=4 tools/train.py -c configs/partial_glint360k.yaml
