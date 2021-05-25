# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import *
from utils.data import *

plt.style.use('ggplot')

# %%
transform_train = transforms.Compose([
    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.4),
    transforms.RandomApply([transforms.RandomAffine(0, scale=(0.8,1.2))], 0.4),
    transforms.RandomApply([transforms.RandomAffine(10)], p=0.4),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5059], [0.2547]),
])
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5059], [0.2547]),
])

# %%
df = pd.read_csv('./data/challengeA_train.csv', index_col=0)[:100]
class_weights = torch.tensor(get_class_weights(df['emotion']))
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print(f'Dataset size: {len(df)}')

batch_size = 32
train_loader, valid_loader, test_loader = get_dataloaders_loopai(df, transform_train, transform_valid, batch_size)

batch_X_train, batch_y_train = next(iter(train_loader))
imshow(utils.make_grid(batch_X_train[:8]), title=[class_names[y] for y in batch_y_train[:4]])
batch_X_train.shape

# %%
df_fer = pd.read_csv('data/fer2013.csv')
df_fer['pixels_array'] = [np.fromstring(x, sep=' ').reshape(48, 48)/255 for x in df_fer['pixels']]
class_weights = torch.tensor(get_class_weights(df_fer['emotion']))
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print(f'Dataset size: {len(df_fer)}')

batch_size = 32
train_loader, valid_loader, test_loader = get_dataloaders_fer2013(df_fer, transform_train, transform_valid, batch_size)

batch_X_train, batch_y_train = next(iter(train_loader))
imshow(utils.make_grid(batch_X_train[:8]))
batch_X_train.shape

# %%
import torch.nn as nn
import torch.optim as optim
from models.dcnn import *
from models.vit import *
from models.resnet import *
from models.densenet import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# net = ResNet101(False).to(device)
net = DCNN2().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(net.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, threshold=1e-3, verbose=True)
net(batch_X_train)

# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
# my_list = ['fc.weight', 'fc.bias']
# head = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))))
# body = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, net.named_parameters()))))
# optimizer = optim.SGD([
#                        {'params': head, 'lr': 3e-3},
#                        {'params': body},
#                       ], lr=3e-4, momentum=0.9, nesterov=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, threshold=1e-3, verbose=True)

# %%
from utils.train import *

repeat = 1
total_train_loss, total_train_acc = [], []
total_valid_loss, total_valid_acc = [], []
epochs = 10
for i in range(repeat):
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
from utils.plot import *
plot_loss_accuracy(total_train_loss, total_valid_loss, total_train_acc, total_valid_acc)
