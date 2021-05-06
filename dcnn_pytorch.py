# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from utils.data import load_data, FaceDataset

plt.style.use('ggplot')

# %%
path_train = './data/images_train'
path_test = './data/images_test'

df = pd.read_csv('./data/challengeA_train.csv', index_col=0)
df_train, df_test = train_test_split(df, test_size=0.2)
df_train, df_valid = train_test_split(df_train, test_size=0.25)

print(f'Train: {len(df_train)}')
print(f'Valid: {len(df_valid)}')
print(f'Test: {len(df_test)}')

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5059], [0.2547])
    # transforms.Resize(224, interpolation=Image.NEAREST)
])

batch_size = 32
train_data = FaceDataset(df_train, path_train, transform=transform)
valid_data = FaceDataset(df_valid, path_train, transform=transform)
test_data = FaceDataset(df_test, path_test, transform=transform)
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0)

batch_X_train, batch_y_train = next(iter(train_loader))
batch_y_train.shape

# %%
psum = torch.tensor([0., 0., 0.])
psum_sq = torch.tensor([0., 0., 0.])

for image, _ in train_loader:
    psum += image.sum(axis=[0,2,3])
    psum_sq += (image**2).sum(axis=[0,2,3])

# mean and std
total_pixel = len(df_train)*48*48
total_mean = psum / total_pixel
total_std = torch.sqrt((psum_sq / total_pixel) - (total_mean ** 2))

# output
print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))

# %%
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
        x = F.softmax(self.fc2(x), dim=1)
        return x

net = DCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=3e-4, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net(batch_X_train[0].unsqueeze(0))

# %%
from utils.model import train
epochs = 10
train_loss, train_acc, valid_loss, valid_acc = train(
                                                net, 
                                                train_loader, 
                                                valid_loader, 
                                                criterion, 
                                                optimizer, 
                                                epochs, 
                                                device
                                            )

# %%
plt.plot(epochs, train_loss, label='train_loss')
plt.plot(epochs, valid_loss, label='valid_loss')
plt.legend()
plt.show()

plt.plot(epochs, train_acc, label='train_acc')
plt.plot(epochs, valid_acc, label='valid_acc')
plt.legend()
plt.show()
