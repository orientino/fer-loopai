"""
Modified version of VGGNet, we use a narrower version where the last block has fewer feature maps,
apply the dropout layer after each block and use a single hidden FC at the end.

original sota vggnet: https://github.com/usef-kh/fer/blob/4655ce611d3f24e66d08ff5f07cba80e68322b1c/models/vgg.py
"""

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VGGNet(nn.Module):
    def __init__(self, drop=0.2):
        super().__init__()

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4a = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(256)
        self.bn4b = nn.BatchNorm2d(256)

#         self.lin1 = nn.Linear(256 * 3 * 3, 4096)
#         self.lin2 = nn.Linear(4096, 4096)
#         self.lin3 = nn.Linear(4096, 7)
        self.lin1 = nn.Linear(256 * 3 * 3, 1024)
        self.lin2 = nn.Linear(1024, 7)

        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.drop(self.pool(x))

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.drop(self.pool(x))

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.drop(self.pool(x))
        
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.drop(self.pool(x))
        
        x = x.view(-1, 256 * 3 * 3)
        x = F.gelu(self.drop(self.lin1(x)))
        x = self.lin2(x)
        
        return x