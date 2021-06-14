from torchvision import models
from torch import nn

def MobileNetV3L(pretrained=False):
    """ResNet  parameters pretrained using ImageNet"""
    model = models.mobilenet_v3_large(pretrained=pretrained, num_classes=7)
    model.features[0][0].weight = nn.Parameter(model.features[0][0].weight.sum(dim=1, keepdim=True))
    return model
