# ================================================================
# Created on Thu Dec 19 2017
#
# @author: congshanzhang
# Shallow Neural Network
# Multiclass Classification
#
# Dataset: Iris Data
# ================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
step 0) load and check dataset 
"""
directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_fully_connected_nnet_intro/Data'
df = pd.read_csv(directory+'/iris.csv')
df.head()

# draw data based on class
import seaborn as sns
sns.pairplot(df, hue="species")  # we can see there are four features and three classes

"""
step 1) get X and y
"""
# get features X
X = df.drop('species', axis=1)
X.head()

target_names = df['species'].unique()
target_names

# build a dictionary to hold target names
# This dictionary defines a map 
target_dict = {n:i for i, n in enumerate(target_names)}
target_dict

# get categorical y
y= df['species'].map(target_dict)
y.head()

# convert y into categorical dummies 
from keras.utils.np_utils import to_categorical
y_cat = to_categorical(y)
y_cat[:10]
"""
out: array([[ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            ..., 
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.],
            [ 1.,  0.,  0.]])
So the output y should be a vector of 1 and 0's, indicating which class each data point belongs.
"""

"""
step 2) training-testing split
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat, test_size=0.2)

"""
step 3) Train a single layer multiclass classification model
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

# build a single layer model
model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='softmax'))
model.compile(Adam(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])

# validation_split will take out 10% of the traning data and treat it as if it is
# a testing dataset and run test on that along each epoch.
model.fit(X_train, y_train, epochs=20, validation_split=0.1)

"""
step 4) Validate predictions
"""
y_pred = model.predict(X_test)
y_pred[:5]

# for each row, return the index of the largest probability.
# Note that in np.argmax, axis = 1 acctually means row-wise
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(y_test_class, y_pred_class))
confusion_matrix(y_test_class, y_pred_class)

