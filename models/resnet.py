# %%
from torchvision import models
from torch import nn

def ResNet50(pretrained):
    """ResNet 23.5M parameters"""
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(2048, 7)
    model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
    return model

def ResNet101(pretrained):
    """ResNet 58.1M parameters"""
    model = models.resnet152(pretrained=pretrained)
    model.fc = nn.Linear(2048, 7)
    return model

def ResNetCustom():
    """ResNet 23.5M parameters"""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 7)
    model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
    return model

# # add a convolution to split input into 3 layers
# x = torch.randn(1, 1, 224, 224)
# model = models.vgg16(pretrained=False) # pretrained=False just for debug reasons
# first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
# first_conv_layer.extend(list(model.features))  
# model.features= nn.Sequential(*first_conv_layer )  
# output = model(x)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 7)
model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1, keepdim=True))
model.eval()