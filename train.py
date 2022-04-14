import timm
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses
from data import FaceData

# model_names = timm.list_models()
# from pprint import pprint
# pprint(model_names)

model = timm.create_model(
    "cspresnet50", pretrained=False, num_classes=0, global_pool=""
)
input_data = torch.randn(1, 3, 224, 224)
print(model(input_data).shape)

loss_func = losses.CircleLoss()

dataset = FaceData(img_root="/dataset/dataset/face_test")
train_dataloader = DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True,
)

for i, (img, label) in enumerate(train_dataloader):
    img = torch.from_numpy(img)
    img = img / 255.
    img = img.cuda()
    label = label.cuda()

