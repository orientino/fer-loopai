import numpy as np
import pandas as pd
import matplotlib.image as img
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_data(path_file, path_data):
    """
    Args
        path_file (str)
        path_data (str)

    Returns
        [n,48,48,3], [n, 7]
    """

    df = pd.read_csv(path_file, sep=',')[:32]
    image_files = df['image_id']
    image_labels = df['emotion']
    X, y = [], []

    for file, label in zip(image_files, image_labels):
        tmp_x = img.imread(path_data + file + '.jpg')
        tmp_y = np.array([1 if label == i else 0 for i in range(7)])
        X.append(tmp_x)
        y.append(tmp_y)

    X = np.array(X).astype(float)
    y = np.array(y).astype(float)
    
    return X, y


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


# class FaceDatasetOld(Dataset):
#     def __init__(self, data, labels, transform=None):
#         super().__init__()
#         self.data = data
#         self.labels = labels
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         image, label = self.data[index], self.labels[index] 
#         image = Image.fromarray(image).convert('RGB')

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, label