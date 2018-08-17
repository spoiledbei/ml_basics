# ================================================================
# Created on Thu Dec 26 2017
#
# @author: congshanzhang
# Convolutional Neural Network
# Images Recognition Classification
# 
# Dataset: CIFAR-10
#
# The goal is to classify 10 different images
#       airplane
#       automobile
#       bird
#       cat
#       deer
#       dog
#       frog
#       horse
#       ship
#       truck
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
step 0) import data
"""
directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_convolutional_nnet_intro'
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
 
X_train.shape # (50000, 32, 32, 3)
y_train.shape

X_train[0]
# show one image
plt.imshow(X_train[0])


"""
step 1) reshape X into 4-dim tensor, y into categorical
"""
# rescale the data by mean image
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train -= np.mean(X_train,axis=0)
X_test -= np.mean(X_test,axis=0)


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
model.add(Conv2D(32, (3, 3), 
                 padding='same',
                 input_shape=(32, 32, 3),
                 activation = 'relu'))   # add a conv layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))  # add a conv layer
model.add(MaxPool2D(pool_size=(2, 2)))  # add a pooling layer

model.add(Conv2D(64, (3, 3),padding='same',activation='relu')) # add a conv layer
model.add(Conv2D(64, (3, 3),activation='relu')) # add a conv layer
model.add(MaxPool2D(pool_size=(2, 2))) # add a pooling layer

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))  # output layer

# compile CNN
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

# train the CNN
model.fit(X_train, y_train_cat, batch_size=32,
          epochs=2, verbose=1, validation_data=(X_test, y_test_cat),shuffle=True)




