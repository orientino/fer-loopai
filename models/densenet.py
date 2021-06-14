from torchvision.models.densenet import densenet121, densenet169
from torchvision import models
from torch import nn


def DenseNet161(pretrained=False):
    """DenseNet 28.6M parameters"""
    model = models.densenet161(pretrained=pretrained)
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.sum(dim=1, keepdim=True))
    return model


def DenseNet169(pretrained=False):
    """DenseNet 14.1M parameters"""
    model = models.densenet169(pretrained=False)
    # model.classifier = nn.Linear(1664, 7)
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.sum(dim=1, keepdim=True))
    return model


def DenseNet121(pretrained=False):
    """DenseNet 7.9M parameters"""
    model = models.densenet121(pretrained=pretrained)
    # model.classifier = nn.Linear(1024, 7)
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.sum(dim=1, keepdim=True))
    return model
