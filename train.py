import timm
import torch
from pprint import pprint
from pytorch_metric_learning import losses

model_names = timm.list_models()
# pprint(model_names)

model = timm.create_model(
    "cspresnet50", pretrained=False, num_classes=0, global_pool=""
)
input_data = torch.randn(1, 3, 224, 224)
print(model(input_data).shape)

loss_func = losses.CircleLoss()
