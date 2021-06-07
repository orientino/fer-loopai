import torch
import pickle
from torchvision import models
from torch import nn

def ResNet18(pretrained=False):
    """ResNet  parameters pretrained using ImageNet"""
    model = models.resnet18(pretrained=pretrained, num_classes=7)
    # model.fc = nn.Linear(512, 7)
    model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
    return model


def ResNet50(pretrained=False):
    """ResNet 23.5M parameters pretrained using ImageNet"""
    model = models.resnet50(pretrained=pretrained)
    # model.fc = nn.Linear(2048, 7)
    model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
    return model


def ResNet50VGGFace2():
    """ResNet 23.5M parameters pretrained using VGGFace2"""
    model = models.resnet50(pretrained=False, num_classes=8631)
    load_state_dict(model, './pretrained/resnet50_ft_weight.pkl')
    model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
    return model


def load_state_dict(model, fname):
    """Pretrained models from https://github.com/cydonia999/VGGFace2-pytorch"""
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

m = ResNet18()
print(m.eval())