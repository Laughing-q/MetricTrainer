from timm import create_model
from pprint import pprint
import timm
import torch
model_names = timm.list_models(pretrained=True)
pprint(model_names)

model = create_model('cspresnet50', pretrained=False, global_pool='', num_classes=0)
model.eval()
o = model(torch.randn(2, 3, 112, 112))
print(o.shape)

# from metric_trainer.models import build_model
# from omegaconf import OmegaConf
# import torch
#
# cfg = OmegaConf.load("configs/test_folder.yaml")
# model = build_model(cfg.MODEL)
# # model.eval()
# img = torch.randn((2, 3, 112, 112))
# output = model(img)
# print(output.shape)
