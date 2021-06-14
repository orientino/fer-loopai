"""
This module is used to to preprocess and clean the dataset by removing the anomalies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import face_recognition
from sklearn.model_selection import train_test_split
from torchvision.transforms.transforms import PILToTensor, ToTensor
from PIL import Image
from torchvision import *
plt.style.use('ggplot')


def stratify_split_loopai(df):
    # split and save the dataset in 80/20 stratified fashion
    df_train, df_valid = train_test_split(df, test_size=0.2, stratify=df['emotion'])
    df_train.to_csv("data/challengeA_train_clean_stratify.csv")
    df_valid.to_csv("data/challengeA_valid_clean_stratify.csv")


def check_duplicates(df):
    images, images_names = [], []
    for i in df['image_id']:
        path = os.path.join('data/images_train', i+'.jpg')
        image = Image.open(path)
        image = ToTensor()(image)
        images.append(image)
        images_names.append(i)

    from collections import Counter
    c = Counter(images)
    duplicates = []
    for i,img in enumerate(c):
        if c[img] > 1:
            duplicates.append(i)

    return duplicates


def check_face(df):
    anomalies, anomalies_names = [], []
    for i in df['image_id']:
        path = os.path.join('data/images_train', i+'.jpg')
        # path = os.path.join('data/images_train', '0bb37430-d2b6-4552-bb2c-5a6576724520'+'.jpg')
        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image)

        image = Image.open(path)
        if (len(face_locations) == 0):
            anomalies.append(ToTensor()(image))
            anomalies_names.append(i)

    return anomalies, anomalies_names


def check_lowstd(df):
    # check for images with low std
    thresh = 0.15
    anomalies, anomalies_names = [], []

    for i in df['image_id']:
        path = os.path.join('data/images_train', i+'.jpg')
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
            anomalies_names.append(i)

        # print('mean: ' + str(total_mean))
        # print('std:  ' + str(total_std))
    return anomalies, anomalies_names
  

def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


df = pd.read_csv('data/challengeA_train.csv')
df.groupby('emotion').count()['image_id'].plot(kind='bar')
len(df)

# visualize the anomalies
anomalies, anomalies_names = check_lowstd(df)
imshow(utils.make_grid(anomalies[:16]))
print(len(anomalies))

# # save the cleaned data
# df_new = df[~df['image_id'].isin(anomalies_names)]
# df_new.to_csv('data/challengeA_train_clean.csv')
# print(len(df))
# print(len(df_new))