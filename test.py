# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import threshold
from torchvision.transforms.transforms import PILToTensor, ToTensor
plt.style.use('ggplot')

df = pd.read_csv('./data/challengeA_train.csv')
train_count = df.groupby('emotion').count()['image_id']
train_count.plot(kind='bar')

# %%
# compute class weights 
# w_j = n_samples / (n_classes * n_samples_j)
n_classes = len(df['emotion'].unique())
class_weight = [len(df)/(n_classes*len(df[df['emotion']==i])) 
                for i in range(n_classes)]
class_weight

# %%
from PIL import Image
import os
import torch
from torchvision import *

thresh = 0.01
anomalies = []
for i in df['image_id'][:]:
    path = os.path.join('data/images_train', i+'.jpg')
    # path = os.path.join('data/images_train', '0bb37430-d2b6-4552-bb2c-5a6576724520'+'.jpg')
    image = Image.open(path)

    # compute mean std
    image = ToTensor()(image)
    psum = image.sum(axis=[1, 2])
    psum_sq = (image**2).sum(axis=[1,2])

    total_pixel = 48*48
    total_mean = psum / total_pixel
    total_std = torch.sqrt((psum_sq / total_pixel) - (total_mean ** 2))

    if (total_std < thresh):
        anomalies.append(image)

    # print('mean: ' + str(total_mean))
    # print('std:  ' + str(total_std))

# %%
def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

imshow(utils.make_grid(anomalies[:32]))
print(len(anomalies))
print(len(df))
print(len(df[~df['image_id'].isin(anomalies)]))

# %%
df[~df['image_id'].isin(anomalies)]