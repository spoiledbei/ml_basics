# ================================================================
# Created on Thu Dec 20 2017
#
# @author: congshanzhang
# Deep Neural Network
# Gradient Descent Project Classification
# Tuning parameters
# 
# Wines datase with features:
#       Class                           178 non-null int64
#       Alcohol                         178 non-null float64
#       Malic_acid                      178 non-null float64
#       Ash                             178 non-null float64
#       Alcalinity_of_ash               178 non-null float64
#       Magnesium                       178 non-null int64
#       Total_phenols                   178 non-null float64
#       Flavanoids                      178 non-null float64
#       Nonflavanoid_phenols            178 non-null float64
#       Proanthocyanins                 178 non-null float64
#       Color_intensity                 178 non-null float64
#       Hue                             178 non-null float64
#       OD280-OD315_of_diluted_wines    178 non-null float64
#       Proline                         178 non-null int64
#
# The goal is to classify artificial 0-1 data
# ================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
step 0) load data
"""
directory = '/Users/congshanzhang/Documents/Office/machine learning/Gradient_Descent'
df = pd.read_csv(directory+'/Data/wines.csv')
df.head()
df.info()
df.describe()


"""
step 1) Visualize Data
"""
# check class. There are 3 classes
df['Class'].value_counts()

# plot the histograms for each feature (Method 1)
plt.figure(figsize=(15, 5))
for i, feature in enumerate(df.columns[1:]): # except the first column
    plt.subplot(4, 4, i+1)
    df[feature].plot(kind='hist', title=feature)
    plt.xlabel(feature)
    
# plot the histograms for each feature (Method 2)
_ = df.hist(figsize=(12,10))    

# For classification problems, explore correlations of features with labels
import seaborn as sns
sns.pairplot(df, hue="Class")

# correlations (heatmap)
sns.heatmap(df.corr(),annot = True)

"""
step 2) scale features and get X, y
"""
# Minmax normalization
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
# To scale the values into the range [0.0, 1.0]
# For each feature, x_scaled = (x - min)/(max-min)
X = mms.fit_transform(df.drop('Class', axis=1))
X.shape

"""
# Alternatively, standard normalization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(df.drop('Class', axis=1))
"""
# use pandas.get_dummies() to change y into categorical variables.
y = df['Class']
y_cat = pd.get_dummies(y).values  

 
"""
step 3) train/test split
"""
from sklearn.model_selection import train_test_split
# use 20% testing set. Set seed to be 22
X_train, X_test, y_train, y_test = train_test_split(X, y_cat,test_size=0.2,random_state=22)


"""
step 4) Build a deep neural network
"""   
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,RMSprop,SGD

K.clear_session() # clear model from memory

# build a 2-layer NN model
model = Sequential()
model.add(Dense(26, input_shape=(X.shape[1],), kernel_initializer='he_normal', activation='relu'))  # 13 features
model.add(Dense(13, activation='relu'))
model.add(Dense(3, activation='softmax'))  # three classes
model.compile(Adam(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train the data and record training history
h = model.fit(X_train, y_train, batch_size=8, epochs=20, verbose = 1, validation_split=0.2)

# visualize training process
historydf = pd.DataFrame(h.history, index=h.epoch)
historydf.plot(ylim=(0,1))


"""
step 5) Build a smaller model and play with learning rate, optimizer, initilizer
"""
K.clear_session() # clear model from memory

# build a 3-layer NN model
model = Sequential()
model.add(Dense(8, input_shape=(X.shape[1],), kernel_initializer='he_normal',activation='tanh'))  # 13 features
model.add(Dense(5, kernel_initializer='he_normal', activation='tanh'))
model.add(Dense(2, kernel_initializer='he_normal',activation='tanh'))
model.add(Dense(3, activation='softmax'))  # three classes
model.compile(Adam(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train the data and record training history
h = model.fit(X_train, y_train, batch_size=16, epochs=20, verbose = 1)


"""
step 6) Define feature function and visualize classification of an inner layer
"""
inp = model.layers[0].input  # input of the 1st layer
out = model.layers[2].output # output of the 3rd layer
features_function = K.function([inp], [out])

# this is the output matrix of N*2 of the 3rd layer                 
features = features_function([X_test])[0]
features.shape

# visualize after transforming the original 13 features into 2 features at the 3rd layer,
# how the data are distributed according to the 2 features.
for i in range(1, 21): # for each epoch
    plt.subplot(5, 4, i)
    h = model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=0) # each time epochs=1. This will build on the previous epochs.
    test_accuracy = model.evaluate(X_test, y_test)[1]
    features = features_function([X_test])[0]
    plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')
    #plt.xlim(-0.5, 3.5)
    #plt.ylim(-0.5, 4.0)
    plt.title('Epoch: {}, Test Acc: {:3.1f} %'.format(i, test_accuracy * 100.0))

plt.tight_layout()



"""
step 7) use Functional API instead of Sequential API
"""
from keras.layers import Input
from keras.models import Model  

K.clear_session()

inputs = Input(shape=(13,)) # we need to define an input layer
x = Dense(8, kernel_initializer='he_normal', activation='tanh')(inputs) # 1st layer as a function of inputs
x = Dense(5, kernel_initializer='he_normal', activation='tanh')(x) # 2nd layer as a function of the output of the previous layer
second_to_last = Dense(2, kernel_initializer='he_normal',activation='tanh')(x)  # from the 3rd to the last layer
outputs = Dense(3, activation='softmax')(second_to_last) # the last layer

model = Model(inputs=inputs, outputs=outputs) # fix together all layers and create a model

model.compile(RMSprop(lr=0.05),
              'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)

features_function = K.function([inputs], [second_to_last])
features = features_function([X_test])[0]
plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')


"""
step 8) use Callbacks: ModelCheckpoint, EarlyStopping, TensorBoard
"""
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

checkpointer = ModelCheckpoint(filepath=directory+"/weights.hdf5",
                               verbose=1, save_best_only=True) # the latest best model will not be overwritten

earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')

tensorboard = TensorBoard(log_dir=directory+'/tensorboard/')


K.clear_session()

inputs = Input(shape=(13,))

x = Dense(8, kernel_initializer='he_normal', activation='tanh')(inputs)
x = Dense(5, kernel_initializer='he_normal', activation='tanh')(x)
second_to_last = Dense(2, kernel_initializer='he_normal',
                       activation='tanh')(x)
outputs = Dense(3, activation='softmax')(second_to_last)

model = Model(inputs=inputs, outputs=outputs)

model.compile(RMSprop(lr=0.05), 'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32,
          epochs=20, verbose=2,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, earlystopper, tensorboard])



