import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.image as img

def load_data(path_file, path_data):
    """
    Args
        path_file (str)
        path_data (str)

    Returns
        [n,48,48,3], [n, 7]
    """

    df = pd.read_csv(path_file, sep=',')
    image_files = df['image_id']
    image_labels = df['emotion']
    X, y = [], []
    n = 7

    for file, label in zip(image_files, image_labels):
        tmp_x = img.imread(path_data + file + '.jpg')
        # tmp_x = cv.imread(path_data + file + '.jpg')
        tmp_y = np.array([1 if label == i else 0 for i in range(n)])
        X.append(tmp_x)
        y.append(tmp_y)

    X = np.array(X).astype(float)
    y = np.array(y).astype(float)
    
    return X, y
