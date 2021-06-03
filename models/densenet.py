from torchvision.models.densenet import densenet121, densenet169
# from utils.model import count_parameters
from torchvision import models
from torch import nn
import torch

def DenseNet161(pretrained):
    """DenseNet 28.6M parameters"""
    model = models.densenet161(pretrained=pretrained)
    model.fc = nn.Linear(2048, 7)
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.sum(dim=1, keepdim=True))
    return model

def DenseNet169(pretrained):
    """DenseNet 14.1M parameters"""
    model = models.densenet169(pretrained=pretrained)
    model.fc = nn.Linear(2048, 7)
    # model.fc = nn.Sequential(
    #     nn.Linear(2048, 7),
    #     nn.ReLU(),
    #     nn.Linear()
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.sum(dim=1, keepdim=True))
    return model

def DenseNet201(pretrained):
    """DenseNet 20.0M parameters"""
    model = models.densenet201(pretrained=pretrained)
    model.fc = nn.Linear(2048, 7)
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.sum(dim=1, keepdim=True))
    return model

def DenseNet121(pretrained):
    """DenseNet 7.9M parameters"""
    model = models.densenet121(pretrained=pretrained)
    model.fc = nn.Linear(2048, 7)
    model.features.conv0.weight = nn.Parameter(model.features.conv0.weight.sum(dim=1, keepdim=True))
    return model