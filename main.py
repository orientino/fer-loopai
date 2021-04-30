# %%
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.utils import to_categorical

# %%
def load_data(path_file, path_data):
    """
    Args
        path_file (str)
        path_data (str)

    Returns
        list(list(int)), list(int)
    """

    test = cv.imread("./data/images_train/0a00ad27-c7b8-4c99-b67a-3cd31290acb9.jpg")
    plt.imshow(test)
    test = np.array(test)
    test.shape
    pass

# %%
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# %%


# %%
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    class_weight = class_weight,
                    epochs=12,
                    batch_size=64)