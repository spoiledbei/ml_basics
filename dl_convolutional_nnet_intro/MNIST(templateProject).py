# ================================================================
# Created on Thu Dec 20 2017
#
# @author: congshanzhang
# Convolutional Neural Network
# Digits Recognition Classification
# 
# Dataset: MNIST
#
# The goal is to classify digits from 0-9
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
step 0) import data
"""
directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_convolutional_nnet_intro'
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data(directory+'/Data/mnist.npz') 

X_train.shape # np.array(60000,28,28) there are 60000 images and each one is 28*28 
X_test.shape  # np.array(10000,28,28)

X_train[0]
# show one image
plt.imshow(X_train[0], cmap='gray')

"""
step 1) reshape X into 4-dim tensor, y into categorical
"""
# rescale the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# rearrange the input size
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train.shape # input tensor with dimension (60000, 28, 28, 1) 

# change y into categorical
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
y_train_cat.shape
y_test_cat.shape


"""
step 2) build CNN
"""
from keras.layers import Conv2D
from keras.layers import MaxPool2D, AvgPool2D
from keras.layers import Flatten, Activation
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

K.clear_session()

# build CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1))) # add a conv layer
model.add(MaxPool2D(pool_size=(2, 2))) # add a pooling layer
model.add(Activation('relu')) # nonlinear transformation

model.add(Conv2D(64, (3, 3))) # add a conv layer
model.add(MaxPool2D(pool_size=(2, 2))) # add a pooling layer
model.add(Activation('relu')) # nonlinear transformation

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # output layer

# compile CNN
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

# train the CNN
model.fit(X_train, y_train_cat, batch_size=128,
          epochs=2, verbose=1, validation_split=0.3)

model.evaluate(X_test, y_test_cat)





