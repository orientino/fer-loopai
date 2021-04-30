# %%
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.utils import to_categorical
plt.style.use('ggplot')

# %%
def load_data(path_file, path_data):
    """
    Args
        path_file (str)
        path_data (str)

    Returns
        list(list(int)), list(int)
    """

    df = pd.read_csv(path_file, sep=',')
    image_files = df['image_id']
    image_labels = df['emotion']
    X, y = [], []
    n = 7

    for file, label in zip(image_files[:50], image_labels):
        tmp_x = cv.imread(path_data + file + '.jpg')
        tmp_y = np.array([1 if label == i else 0 for i in range(n)])
        X.append(tmp_x)
        y.append(tmp_y)

    X = np.array(X)
    y = np.array(y)
    
    return X, y

# %%
X, y = load_data('./data/challengeA_train.csv', './data/images_train/')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X.shape
# y.shape
# plt.imshow(X[0])

# %%
model = models.Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    # class_weight = class_weight,
                    epochs=100,
                    batch_size=32)

loss_train = history.history['loss']
loss_valid = history.history['val_loss']
acc_train = history.history['accuracy']
acc_valid = history.history['val_accuracy']
epochs = range(1, len(loss_train)+1)

# %%
# plot loss 
plt.plot(epochs, loss_train, label='loss_train')
plt.plot(epochs, loss_valid, label='loss_valid')
plt.legend()
plt.show()

# %%
# plot accuracy
plt.plot(epochs, acc_train, label='acc_train')
plt.plot(epochs, acc_valid, label='acc_valid')
plt.legend()
plt.show()
