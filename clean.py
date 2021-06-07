# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from torchvision.transforms.transforms import PILToTensor, ToTensor
plt.style.use('ggplot')

df = pd.read_csv('data/challengeA_train_clean.csv')
len(df)
# train_count = df.groupby('emotion').count()['image_id']
# train_count.plot(kind='bar')
# len(df)

# %%
from PIL import Image
from torchvision import *
import os
import face_recognition
import cv2 as cv


# # check for duplicates
# images, images_names = [], []
# for i in df['image_id']:
#     path = os.path.join('data/images_train', i+'.jpg')
#     image = Image.open(path)
#     image = ToTensor()(image)
#     images.append(image)
#     images_names.append(i)

# from collections import Counter
# c = Counter(images)

# duplicates = []
# for i,img in enumerate(c):
#     if c[img] > 1:
#         duplicates.append(i)

# print(len(duplicates))


# # check for images with no face
# for i in df['image_id']:
#     path = os.path.join('data/images_train', i+'.jpg')
#     # path = os.path.join('data/images_train', '0bb37430-d2b6-4552-bb2c-5a6576724520'+'.jpg')
#     image = face_recognition.load_image_file(path)
#     face_locations = face_recognition.face_locations(image)
#     # print(face_locations)

#     image = Image.open(path)
#     if (len(face_locations) == 0):
#         anomalies.append(ToTensor()(image))
#         anomalies_names.append(i)


# # check for images with low std
# thresh = 0.15
# anomalies, anomalies_names = [], []

# for i in df['image_id']:
#     path = os.path.join('data/images_train', i+'.jpg')
#     image = Image.open(path)

#     # compute mean std
#     image = ToTensor()(image)
#     psum = image.sum(axis=[1, 2])
#     psum_sq = (image**2).sum(axis=[1,2])

#     total_pixel = 48*48
#     total_mean = psum / total_pixel
#     total_std = torch.sqrt((psum_sq / total_pixel) - (total_mean ** 2))

#     if (total_std < thresh):
#         anomalies.append(image)
#         anomalies_names.append(i)

    # print('mean: ' + str(total_mean))
    # print('std:  ' + str(total_std))

# %%
def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

imshow(utils.make_grid(anomalies[1000:1200]))
print(len(anomalies))

# %%
# m = [9, 12, 29, 35]

# %%
df = pd.read_csv('data/challengeA_train.csv', index_col=0)
df_new = df[~df['image_id'].isin(anomalies_names)]
df_new.to_csv('data/challengeA_train_clean.csv')
print(len(df))
print(len(df_new))

# %%
df = pd.read_csv('data/challengeA_train.csv', index_col=0)
dfc = pd.read_csv('data/challengeA_train_clean.csv', index_col=0)

print(len(df))
print(len(dfc))
