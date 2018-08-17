# ================================================================
# Created on Thu Dec 19 2017
#
# @author: congshanzhang
# Deep Neural Network
# Binary Classification
# 
# Multilayer logistic model
#
# The goal is to classify artificial 0-1 data
# ================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
step 0) load data
"""
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.legend(['0', '1'])

X.shape

"""
step 1) train a simply logistic model as benchmark
"""
from sklearn.model_selection import train_test_split

# use 30% testing set. Set seed to be 42
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

# build a single layer logistic model
model = Sequential()
model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, verbose=0)

# evaluate model on testing set
results = model.evaluate(X_test, y_test)

results # two numbers: loss and precision

print("The Accuracy score on the Train set is:\t{:0.3f}".format(results[1]))

# plot decision boundary
def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    c = model.predict(ab)
    cc = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.2)
    plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
    plt.legend(['0', '1'])
    
plot_decision_boundary(model, X, y)
# ===================================================================
# Note: In one layer logistic model, the decision boundary is linear
# ===================================================================

"""
step 2) Train a deep neural network model
"""
# add 3 dense layers
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='tanh'))  # 4 nodes in the first layer
model.add(Dense(2, activation='tanh'))  # 2 nodes in the second layer
model.add(Dense(1, activation='sigmoid'))  # 1 nodes in the final layer
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

# train the neural network
model.fit(X_train, y_train, epochs=100, verbose=0)

# evaluate model on testing set
model.evaluate(X_test, y_test)

from sklearn.metrics import accuracy_score, confusion_matrix

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

print("The Accuracy score on the Train set is:\t{:0.3f}".format(accuracy_score(y_train, y_train_pred)))
print("The Accuracy score on the Test set is:\t{:0.3f}".format(accuracy_score(y_test, y_test_pred)))

# plot decision boundary
plot_decision_boundary(model, X, y)



