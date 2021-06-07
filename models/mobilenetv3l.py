import torch
from torchvision import models
from torch import nn

def MobileNetV3L(pretrained=False):
    """ResNet  parameters pretrained using ImageNet"""
    model = models.mobilenet_v3_large(pretrained=pretrained, num_classes=7)
    # model.classifier[3] = nn.Linear(512, 7)
    model.features[0][0].weight = nn.Parameter(model.features[0][0].weight.sum(dim=1, keepdim=True))
    return model

m = MobileNetV3L()
print(m.eval())
x = torch.rand([1, 1, 48, 48])
y = m(x)
print(y.shape)