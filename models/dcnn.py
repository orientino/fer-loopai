import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNN(nn.Module): 
    def __init__(self):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*12*12, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DCNN1(nn.Module): 
    """19.8M parameters"""
    def __init__(self):
        super(DCNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=0)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.norm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*24*24, 128)
        self.norm4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.norm5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 7)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.drop(self.norm1(F.gelu(self.conv1(x))))
        x = self.drop(self.norm2(F.gelu(self.conv2(x))))
        x = self.drop(F.max_pool2d(self.norm3(F.gelu(self.conv3(x))), 2))
        x = torch.flatten(x, 1)
        x = self.drop(self.norm4(F.gelu(self.fc1(x))))
        x = self.drop(self.norm5(F.gelu(self.fc2(x))))
        x = self.fc3(x)
        return x


class DCNN2(nn.Module): 
    """19.8M parameters"""
    def __init__(self):
        super(DCNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=0)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.norm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*24*24, 128)
        self.norm4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.norm5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 7)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.drop(self.norm1(F.relu(self.conv1(x))))
        x = self.drop(self.norm2(F.relu(self.conv2(x))))
        x = self.drop(F.max_pool2d(self.norm3(F.relu(self.conv3(x))), 2))
        x = torch.flatten(x, 1)

        x = self.drop(F.relu(self.norm4(self.fc1(x))))
        x = self.drop(F.relu(self.norm5(self.fc2(x))))
        x = self.fc3(x)
        return x
