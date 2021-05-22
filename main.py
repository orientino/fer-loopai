# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms.transforms import RandomRotation
from PIL import Image
from utils.data import *

plt.style.use('ggplot')

# %%
path_train = './data/images_train'
path_test = './data/images_test'

# TODO stratify the splitting
df = pd.read_csv('./data/challengeA_train.csv', index_col=0)[:100]
df_train, df_test = train_test_split(df, test_size=0.2)
df_train, df_valid = train_test_split(df_train, test_size=0.25)

print(f'Train: {len(df_train)}')
print(f'Valid: {len(df_valid)}')
print(f'Test: {len(df_test)}')

# compute class weights
class_weights = torch.tensor(get_class_weights(df['emotion']))
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# %%
mean, std = [0.5059], [0.2547]
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    # transforms.Resize(224, interpolation=Image.NEAREST)
])

batch_size = 32
train_data = FaceDataset(df_train, path_train, transform=transform)
valid_data = FaceDataset(df_valid, path_train, transform=transform)
test_data = FaceDataset(df_test, path_test, transform=transform)
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0)

def imshow(img, title=None):
    img = img * std[0] + mean[0]     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

batch_X_train, batch_y_train = next(iter(train_loader))
imshow(utils.make_grid(batch_X_train[:4]), title=[class_names[y] for y in batch_y_train[:4]])
batch_X_train.shape
# print(' '.join('%5s' % batch_y_train[j] for j in range(batch_size)))

# %%
import torch.nn as nn
import torch.optim as optim
from models.dcnn import DCNN, DCNN1
from models.vit import ViT
from models.resnet import *
from models.densenet import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# net = ResNet101(False).to(device)
net = DCNN1().to(device)
print(count_parameters(net))
# net = ViT().to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
# optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, threshold=1e-3, verbose=True)
# net(batch_X_train[0].unsqueeze(0))

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
my_list = ['fc.weight', 'fc.bias']
head = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))))
body = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, net.named_parameters()))))
optimizer = optim.SGD([
                       {'params': head, 'lr': 3e-3},
                       {'params': body},
                      ], lr=3e-4, momentum=0.9, nesterov=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, threshold=1e-3, verbose=True)

# %%
net(batch_X_train)

# %%
from utils.model import *

times = 1
total_train_loss, total_train_acc = [], []
total_valid_loss, total_valid_acc = [], []
epochs = 20
for i in range(times):
    train_loss, train_acc, valid_loss, valid_acc = train(
                                                    net, 
                                                    train_loader, 
                                                    valid_loader, 
                                                    criterion, 
                                                    optimizer,
                                                    scheduler, 
                                                    epochs, 
                                                    device,
                                                )   

    total_train_loss.append(train_loss)
    total_train_acc.append(train_acc)
    total_valid_loss.append(valid_loss)
    total_valid_acc.append(valid_acc)

# %%
from utils.model import *
plot_loss_accuracy(total_train_loss, total_valid_loss, total_train_acc, total_valid_acc)
