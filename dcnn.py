# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.model_selection import train_test_split
from utils.data import load_data

from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.utils import to_categorical
plt.style.use('ggplot')

# %%
X, y = load_data('./data/challengeA_train.csv', './data/images_train/')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X.shape

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
                    epochs=3,
                    batch_size=32)

loss_train = history.history['loss']
loss_valid = history.history['val_loss']
acc_train = history.history['accuracy']
acc_valid = history.history['val_accuracy']
epochs = range(1, len(loss_train)+1)

# %%
# plot loss accuracy
plt.plot(epochs, loss_train, label='loss_train')
plt.plot(epochs, loss_valid, label='loss_valid')
plt.legend()
plt.show()

plt.plot(epochs, acc_train, label='acc_train')
plt.plot(epochs, acc_valid, label='acc_valid')
plt.legend()
plt.show()
