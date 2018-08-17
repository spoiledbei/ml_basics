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

X_test.shape # np.array(10000,28,28)

X_train[0]
# show one image
plt.imshow(X_train[0], cmap='gray')


"""
step 1) Reshape image data and clean data
"""
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

X_train.shape  # np.array(60000,784)

# rescale the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

X_train[0]

# change y into categorical
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
y_train_cat.shape
y_test_cat.shape


"""
step 3) build a fully connected deep NN to classify the images
"""
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

K.clear_session()

model = Sequential()
model.add(Dense(512, input_dim=28*28, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# train model
h = model.fit(X_train, y_train_cat, batch_size=128, epochs=10, verbose=1, validation_split=0.3)

# visualize training process
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')

# test accuracy
test_accuracy = model.evaluate(X_test, y_test_cat)[1]
test_accuracy

"""
step 4) Image filters with convolutions
"""
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from scipy import misc

img = misc.ascent()

img.shape

plt.imshow(img, cmap='gray')

h_kernel = np.array([[ 1,  2,  1],
                     [ 0,  0,  0],
                     [-1, -2, -1]])

plt.imshow(h_kernel, cmap='gray')

res = convolve2d(img, h_kernel)

plt.imshow(res, cmap='gray')


# ===========================================
#
#     Convolutional Neural Networks
#         1. basics
#
# ===========================================
from keras.layers import Conv2D

img.shape

plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')

# reshape into a image tensor (# of images, height, width, # of chanels)
img_tensor = img.reshape((1, 512, 512, 1))

# build a convolutional layer
model = Sequential()
model.add(Conv2D(1, (3, 3), strides=(2,1), input_shape=(512, 512, 1))) 
model.compile('adam', 'mse')

# a feed-forward path with a random kernel filter 
# (note that the model has not been trained yet) 
img_pred_tensor = model.predict(img_tensor) 
img_pred_tensor.shape # (1,255,510,1)

# get the image part
img_pred = img_pred_tensor[0, :, :, 0]  # interior part is the image
plt.imshow(img_pred, cmap='gray')

# find filter 
weights = model.get_weights()
weights[0].shape # (3,3,1,1) height, width, # chanels in input, # chanels in output
plt.imshow(weights[0][:, :, 0, 0], cmap='gray') # show kernel filter

# try a new filter
weights[0] = np.ones(weights[0].shape)
model.set_weights(weights)

# a new feed-forward path
img_pred_tensor = model.predict(img_tensor)

# get and show image
img_pred = img_pred_tensor[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')

# the argument padding = 'same' will add 0's to the four edges.
model = Sequential()
model.add(Conv2D(1, (3, 3), input_shape=(512, 512, 1), padding='same'))
model.compile('adam', 'mse')

img_pred_tensor = model.predict(img_tensor)
img_pred_tensor.shape # (1, 512, 512, 1) notice now that the image is still 512*512


# ========================================================
#  2. Pooling layers
#       to reduce size of image and preserve information
# ========================================================
from keras.layers import MaxPool2D, AvgPool2D

# build a pooling layer with max pooling
model = Sequential()
model.add(MaxPool2D((5, 5), input_shape=(512, 512, 1)))
model.compile('adam', 'mse')

# get and display prediction
img_pred = model.predict(img_tensor)[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')

# build a pooling layer with average pooling
model = Sequential()
model.add(AvgPool2D((5, 5), input_shape=(512, 512, 1)))
model.compile('adam', 'mse')

# get and display prediction
img_pred = model.predict(img_tensor)[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')


# ==============================================================
#  3. Build a CNN
#      3.1. the architecture:
#           conv--pooling--conv--pooling--fulling connected NN
#           The convolutional part handles feature extraction.
#           The fulling connected NN handles the classification.
# ==============================================================
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train.shape # input tensor with dimension (60000, 28, 28, 1) 

from keras.layers import Flatten, Activation

K.clear_session()

# build CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1))) # add a conv layer
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











