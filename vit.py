# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.model_selection import train_test_split
from load_data import load_data
plt.style.use('ggplot')

X, y = load_data('./data/challengeA_train.csv', './data/images_train/')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)
print(f'Train: {X_train.shape}')
print(f'Valid: {X_valid.shape}')
print(f'Test: {X_test.shape}')

# pd.DataFrame(y_train).sum().apply(lambda x: x/len(y_train)).plot()
# pd.DataFrame(y_valid).sum().apply(lambda x: x/len(y_valid)).plot()
# pd.DataFrame(y_test).sum().apply(lambda x: x/len(y_test)).plot()

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

vit = timm.create_model('vit_base_patch16_224', pretrained=True, )
vit.head = nn.Linear(768,7)
# vit.eval()

# %% Transformation using timm (?)
# config = resolve_data_config({}, model=vit)
# transform = create_transform(**config)

# x = Image.fromarray(X_train[0]).convert('RGB')
# tensor = transform(x).unsqueeze(0)
# vit(tensor)

# display(x)
# display(transforms.ToPILImage()(tensor.squeeze()))
# plt.imshow(tensor.squeeze())
# tensor.shape

# from pprint import pprint
# model_names = timm.list_models('*vit*')
# pprint(model_names)

# %%
class FaceDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image, label = self.data[index], self.labels[index] 
        image = Image.fromarray(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, interpolation=Image.NEAREST)
])

batch_size = 32
train_data = FaceDataset(X_train, y_train, transform=transform)
valid_data = FaceDataset(X_valid, y_valid, transform=transform)
test_data = FaceDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0)

batch_X_train, batch_y_train = next(iter(train_loader))
batch_X_train.shape

# for i in range(5):
#     display(transforms.ToPILImage()(batch_X_train[i]))

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# %%
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
vit.head = nn.Linear(768,7)
vit = vit.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vit.parameters(), lr=0.0001, momentum=0.9)

# %%
n_epochs = 5
valid_loss_min = np.Inf
valid_loss, valid_acc = [], []
train_loss, train_acc = [], []

for epoch in range(1, n_epochs+1):
    loss_epoch = 0.0
    correct, total = 0, 0
    print(f'Epoch {epoch}')

    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        
        outputs = vit(input)
        _, label = torch.max(label, dim=1)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        total += label.size(0)
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==label).item()
        
        if (batch_idx) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, len(train_loader), loss.item()))

    train_acc.append(correct / total)
    train_loss.append(loss_epoch / len(train_loader))
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(correct / total):.4f}')
    
    # validation phase
    loss_epoch = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        # net.eval()
        for input, label in (valid_loader):
            input, label = input.to(device), label.to(device)
            outputs = vit(input)
            _, label = torch.max(label, dim=1)

            loss = criterion(outputs, label)

            loss_epoch += loss.item()
            total += label.size(0)
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==label).item()
            
        valid_acc.append(correct / total)
        valid_loss.append(loss_epoch / len(valid_loader))
        network_learned = loss_epoch < validid_loss_min
        print(f'valididation loss: {np.mean(valid_loss):.4f}, valididation acc: {(correct / total):.4f}')
        
        if network_learned:
            validid_loss_min = loss_epoch
            torch.save(vit.state_dict(), f'./vit_base_patch16_224_ep{epoch}_lr1e-3.pt')
            print('Improvement-Detected, save-model')
    # net.train()