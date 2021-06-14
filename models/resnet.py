from torchvision import models
from torch import nn

def ResNet18(pretrained=False):
    """ResNet  parameters pretrained using ImageNet"""
    model = models.resnet18(pretrained=pretrained, num_classes=7)
    model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
    return model


def ResNet50(pretrained=False):
    """ResNet 23.5M parameters pretrained using ImageNet"""
    model = models.resnet50(pretrained=pretrained, num_classes=7)
    model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
    return model
