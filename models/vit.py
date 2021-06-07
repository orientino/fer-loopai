import timm
import torch
from torch import nn
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transfor

class ViT(nn.Module): 
    def __init__(self):
        super(ViT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=48, in_chans=1) 
        self.freeze()
        self.model.head = nn.Linear(768, 7)

    def forward(self, x):
        x = self.model(x)
        return x 

    def unfreeze(self):
        # unfreeze all parameters 
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self):
        # freeze all parameters 
        for param in self.model.parameters():
            param.requires_grad = False