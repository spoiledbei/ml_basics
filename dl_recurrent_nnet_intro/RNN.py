# ================================================================
# Created on Thu Dec 20 2017
#
# @author: congshanzhang
# Recurrent Neural Network
# Time-Series 
# 
# Dataset: 
#
# The goal is to classify digits from 0-9
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_recurrent_nnet_intro/Data'

df = pd.read_csv(directory+'/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows=6, skipfooter=9,
                 engine='python')
df.head()


# ========================
#  Time series prediction
# ========================
from pandas.tseries.offsets import MonthEnd

# change date into format '1991-01-31'
df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')
df.head()

# plot the time-series
df.plot()

# train-test split
split_date = pd.Timestamp('01-01-2011')
train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[split_date:, ['Unadjusted']]

# plot the training and testing parts
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])

# min-max scale the time-series
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)   # fit_transform the training series
test_sc = sc.transform(test)         # transform the testing series

# our object is 1-step prediction 
X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]


# ==========================================================
#   Fully connected predictor (bad model for time series)
# ==========================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras.backend as K

K.clear_session()

model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train, y_train, epochs=200,
          batch_size=2, verbose=1,
          callbacks=[early_stop])

# model testing
y_pred = model.predict(X_test)

# plot the predictions and true values
plt.plot(y_test)
plt.plot(y_pred)

# ==================================================
#   Recurrent predictor (not good for time series)
# ==================================================
from keras.layers import LSTM

X_train.shape # (239,1)

#3D tensor with shape (batch_size, timesteps, input_dim)
X_train[:, None].shape # (239,1,1)

X_train_t = X_train[:, None]
X_test_t = X_test[:, None]


K.clear_session()

model = Sequential()
model.add(LSTM(6, input_shape=(1, 1)))   # input_shape = (1 timestep, 1 number)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train_t, y_train,
          epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)



# ==============================================================================
#  Windows
#       The goal is to use the past 12 points to predict 1 step into the future
# ==============================================================================
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

# ================================
#  Fully Connected on Windows
# ================================
K.clear_session()

model = Sequential()
model.add(Dense(12, input_dim=12, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(X_train, y_train, epochs=200,
          batch_size=1, verbose=1, callbacks=[early_stop])

y_pred = model.predict(X_test)
plt.plot(y_test)
plt.plot(y_pred)

# ================================
#  LSTM on Windows
# ================================
# reshape X into a tensor
X_train_t = X_train.reshape(X_train.shape[0], 1, 12)  #(228, 1 timestep, 12 number)
X_test_t = X_test.reshape(X_test.shape[0], 1, 12)

K.clear_session()
model = Sequential()

model.add(LSTM(6, input_shape=(1, 12)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

model.fit(X_train_t, y_train, epochs=100,
          batch_size=1, verbose=1, callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)






