"""
Main module to train and evaluate the single models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import *
from utils.data import *
from utils.model import *
from utils.train import *
from utils.infer import *
from utils.plot import *
plt.style.use('ggplot')

# data augmentation
transform_train = transforms.Compose([
    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.4),
    transforms.RandomApply([transforms.RandomAffine(0, scale=(0.8,1.2))], 0.4),
    transforms.RandomApply([transforms.RandomAffine(10)], p=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.5059], [0.2547]),
])
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5059], [0.2547]),
])
batch_size = 8

# load dataset
path = './data/'
df_train = pd.read_csv(os.path.join(path, 'challengeA_train_clean_stratify.csv'), index_col=0)[:300] # testing only
df_valid = pd.read_csv(os.path.join(path, 'challengeA_valid_clean_stratify.csv'), index_col=0)[:100] # testing only
df_test = pd.read_csv(os.path.join(path, 'challengeA_test.csv'), index_col=0)
class_weights = torch.tensor(get_class_weights(df_train['emotion']))
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

train_loader, valid_loader, _ = get_dataloaders_loopai(df_train, df_valid, df_test, path, transform_train, transform_valid, batch_size)

import torch.nn as nn
import torch.optim as optim
from models.dcnn import *
from models.vit import *
from models.densenet import *
from models.vggnet import *
from models.resnet_narrow_nocp import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VGGNet().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, threshold=1e-3, verbose=True)

# # different training rules for body and head
# my_list = ['fc.weight', 'fc.bias']
# head = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))))
# body = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, net.named_parameters()))))
# optimizer = optim.SGD([
#                        {'params': head, 'lr': 3e-3},
#                        {'params': body},
#                       ], lr=3e-4, momentum=0.9, nesterov=True)

# training
total_train_loss, total_train_acc = [], []
total_valid_loss, total_valid_acc = [], []
epochs = 10
repeat = 1
for i in range(repeat):
    train_loss, train_acc, valid_loss, valid_acc = fit(
                                                    model, 
                                                    train_loader, 
                                                    valid_loader, 
                                                    criterion, 
                                                    optimizer,
                                                    scheduler, 
                                                    epochs, 
                                                    device,
                                                    save_after=0
                                                )

    total_train_loss.append(train_loss)
    total_train_acc.append(train_acc)
    total_valid_loss.append(valid_loss)
    total_valid_acc.append(valid_acc)

plot_loss_accuracy(total_train_loss, total_valid_loss, total_train_acc, total_valid_acc)

# # after training, load and evaluate the model
# input, label = next(iter(valid_loader))
# load("VGG4.0_6_0.691.tar", model, device=device)

# # plot inference
# _, pred_p, pred = infer(model, input, device)
# plot_infer(input, label, pred_p, pred, class_names)

# # plot saliency map of the first input
# plot_saliency_map(input[6], model)

# # accuracy, P, R, F1, confusion matrix
# evaluate(model, valid_loader, criterion, class_names, device)
