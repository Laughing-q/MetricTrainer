from timm import create_model
import torch.nn as nn
import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)

model = create_model('cspresnet50', pretrained=True, act_layer=nn.SiLU)
model.eval()
