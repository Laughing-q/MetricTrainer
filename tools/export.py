import timm
import torch
from torch import nn
from pprint import pprint


def torch2Onnx(model, dynamic=False):
    """
    pytorch转onnx
    """
    # 输入placeholder
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_output = model(dummy_input)
    print(dummy_output.shape)

    # Export to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        "face.onnx",
        input_names=["inputs"],
        output_names=["outputs"],
        # verbose=False,
        opset_version=12,
        dynamic_axes={
            "inputs": {0: "batch"},  # shape(1,3,640,640)
            "outputs": {0: "batch", 1: "width"},  # shape(1,3,640,640)
        }
        if dynamic
        else None,
    )


# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

img = torch.randn((1, 3, 224, 224))
model = timm.create_model(
    model_name="convnext_base_in22ft1k",
    exportable=True,
    num_classes=0,
    pretrained=True,
    global_pool="",
    norm_layer=nn.BatchNorm2d,
    conv_mlp=True,
    # head_norm_first=True,
    act_layer=nn.SiLU,
)
print(model.num_features)
model.eval()
output = model(img)
print(output.shape)
# torch2Onnx(model)
