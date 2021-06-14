"""
This module contains classes and functions to handle the processing 
of the data for the training of the model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, data, directory, transform=None):
        super().__init__()
        self.data = data.reset_index()
        self.directory = directory
        self.transform = transform
        
    def __len__(self):
      return len(self.data)
    
    def __getitem__(self, index):
        path = os.path.join(self.directory, self.data['image_id'][index]+'.jpg')
        image = Image.open(path)
        label = self.data['emotion'][index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class Fer2013Dataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.data = data.reset_index()
        self.transform = transform
        
    def __len__(self):
      return len(self.data)
    
    def __getitem__(self, index):
        image = self.data['pixels_array'][index]
        label = self.data['emotion'][index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders_loopai(df_train, df_valid, df_test, path, transform_train, transform_valid, batch_size):
    path_train = os.path.join(path, 'images_train')
    path_test = os.path.join(path, 'images_test')

    train_data = FaceDataset(df_train, path_train, transform=transform_train)
    valid_data = FaceDataset(df_valid, path_train, transform=transform_valid)
    test_data = FaceDataset(df_test, path_test, transform=transform_valid)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader


def get_dataloaders_fer2013(df, transform_train, transform_valid, batch_size, test=False):
    df_train = df[df['Usage']=='Training']
    df_valid = df[df['Usage']=='PrivateTest']
    df_test = df[df['Usage']=='PublicTest']

    if test:
        df_train = df[df['Usage']=='Training'][:100]
        df_valid = df[df['Usage']=='PrivateTest'][:20]
        df_test = df[df['Usage']=='PublicTest'][:20]

    train_data = Fer2013Dataset(df_train, transform=transform_train)
    valid_data = Fer2013Dataset(df_valid, transform=transform_valid)
    test_data = Fer2013Dataset(df_test, transform=transform_valid)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader


def get_class_weights(df_label):
    """
        Gives more weight to minority classes.
        The weight for a class is computed as:
            w_j = n_samples / (n_classes * n_samples_j)
        
        The result will be used as an argument in the loss function.
    """
    n_classes = len(df_label.unique())
    class_weights = [len(df_label)/(n_classes*len(df_label[df_label==i])) 
                    for i in range(n_classes)]
    return class_weights


def get_data_mean_std(train_loader, valid_loader, n_samples):
    """
        Compute the mean and std for a dataset.
        Used to normalize the dataset.
    """
    psum = torch.tensor([0., 0., 0.])
    psum_sq = torch.tensor([0., 0., 0.])

    for image, _ in train_loader:
        psum += image.sum(axis=[0,2,3])
        psum_sq += (image**2).sum(axis=[0,2,3])
    for image, _ in valid_loader:
        psum += image.sum(axis=[0,2,3])
        psum_sq += (image**2).sum(axis=[0,2,3])

    # mean and std
    total_pixel = n_samples*48*48
    total_mean = psum / total_pixel
    total_std = torch.sqrt((psum_sq / total_pixel) - (total_mean ** 2))
    return total_mean, total_std


def imshow(img, title=None):
    mean, std = [0.5059], [0.2547]
    img = img * std[0] + mean[0]     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()