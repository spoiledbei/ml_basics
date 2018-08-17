# ================================================================
# Created on Thu Dec 20 2017
#
# @author: congshanzhang
# Recurrent Neural Network 
# 
# Dataset: Canadian data and MNIST
#
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_recurrent_nnet_intro/Data'
directory2 = '/Users/congshanzhang/Documents/Office/machine learning/Convolutional_neural_nets/Data'

"""
step 0) read data
"""
df = pd.read_csv(directory+'/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows=6, skipfooter=9,
                 engine='python')
df.head()

"""
step 1) train-test split 
"""
from pandas.tseries.offsets import MonthEnd

# change date into format '1991-01-31'
df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')
df.head()

# plot the time-series
df.plot()

# train-test split
split_date = pd.Timestamp('01-01-2011')
train = df.ix[:split_date, ['Unadjusted']]
test = df.ix[split_date:, ['Unadjusted']]

# plot the training and testing parts
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])

"""
step 2) min-max scale the time-series
"""
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)   # fit_transform the training series
test_sc = sc.transform(test)         # transform the testing series


"""
step 3) get X and y
        with 12-step window to predict one-step ahead value
"""
train_sc.shape  # train_cs is np.array (240,1)

train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)
train_sc_df.head()

# window size = 12
for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
    
train_sc_df.head(13)

X_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]

X_test = test_sc_df.dropna().drop('Scaled', axis=1)
y_test = test_sc_df.dropna()[['Scaled']]

X_train.head() # (228, 12)

# get testing and training data
X_train = X_train.values
X_test= X_test.values

y_train = y_train.values
y_test = y_test.values

X_train.shape


"""
step 4)  LSTM on Windows
"""
# reshape X into a tensor (T,12,1)
# This means we consider each input window as a sequence of 12 values 
# that we will pass in sequence to the LSTM
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)  #(228, 12 timestep, 1 scaler)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
import keras.backend as K

K.clear_session()

model = Sequential()
model.add(LSTM(6, input_shape=(12, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train_t, y_train, epochs=100,
          batch_size=32, verbose=1, callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)


"""
Section 2. Use LSTM to do image recognition
"""
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

X_train.shape

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)

print(X_train.shape)
print(X_test.shape)
print(y_train_cat.shape)
print(y_test_cat.shape)

# define RNN model
K.clear_session()
model = Sequential()
model.add(LSTM(32, input_shape=X_train.shape[1:]))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train_cat,
          batch_size=32,
          epochs=100,
          validation_split=0.3,
          shuffle=True,
          verbose=2,
          callbacks=[early_stop]
          )

model.evaluate(X_test, y_test_cat)

