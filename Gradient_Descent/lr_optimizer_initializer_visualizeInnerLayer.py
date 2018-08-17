# =============================================================================
# Created on Thu Dec 22 2017
#
# @author: congshanzhang
# Gradient Descent
# Tasks:
#   Tune learning rate.
#   Tune batch sizes in stochastic gradient descent.
#   Choose different optimizer 
#   Choose initializer for the (random) weights of Keras layers and the bias vector
#   Visualize training process for different parameters.
# 
# Dataset: fake money notes
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

directory = '/Users/congshanzhang/Documents/Office/machine learning/Gradient_Descent/Data/'

df = pd.read_csv(directory+'banknotes.csv')
df.head()
df.info()
df.describe() 

"""
step 1) Visualize Data
"""
# see the classes
df['class'].value_counts()

# For classification problems, explore correlations of features with labels
import seaborn as sns
sns.pairplot(df, hue="class")

# correlations (heatmap)
sns.heatmap(df.corr(),annot = True)

"""
step 2) Baseline Model 1: Random Forest
"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale

X = scale(df.drop('class', axis=1).values)   # This is the same as StandardScaler in sklearn.preprocessing
y = df['class'].values
      
model = RandomForestClassifier()
cross_val_score(model, X, y) # This is a 3 fold cross-validation by default

"""
step 3) Baseline Model 2: Logistic Regression
""" 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD 

K.clear_session() # clear model from memory

model = Sequential()
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# train model
history = model.fit(X_train, y_train,epochs = 10) # record history of training progress
result = model.evaluate(X_test, y_test)            

# visualize the training process
historydf = pd.DataFrame(history.history, index=history.epoch)
historydf.plot(ylim=(0,1))
plt.title("Test accuracy: {:3.1f} %".format(result[1]*100), fontsize=15)


# ===================================
#   manually tune learning rate
# ===================================
dflist = []

learning_rates = [0.01, 0.05, 0.1, 0.5]

for lr in learning_rates:

    K.clear_session()

    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=lr),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))# this is a list of dataframe
    
historydf = pd.concat(dflist, axis=1)    
historydf    
               

# change header of historydf using multi index technique
metrics_reported = dflist[0].columns 
idx = pd.MultiIndex.from_product([learning_rates, metrics_reported],
                                 names=['learning_rate', 'metric'])

historydf.columns = idx
historydf

# plot the training process for different learning rate
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax) # level is the multiindex level
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()



# ===================================
#   manually tune Batch Sizes
# ===================================
dflist = []

batch_sizes = [16, 32, 64, 128]

for batch_size in batch_sizes:
    K.clear_session()

    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, epochs = 10, batch_size=batch_size, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))

historydf = pd.concat(dflist, axis=1)
historydf 

# change header of historydf using multi index technique
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([batch_sizes, metrics_reported],
                                 names=['batch_size', 'metric'])
historydf.columns = idx
historydf

# plot the training process for different batch sizes
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()


# ===================================
#   choose different optimizer
# ===================================
from keras.optimizers import SGD, Adam, Adagrad, RMSprop

dflist = []

optimizers = ['SGD(lr=0.01)',
              'SGD(lr=0.01, momentum=0.3)',
              'SGD(lr=0.01, momentum=0.3, nesterov=True)',  
              'Adam(lr=0.01)',
              'Adagrad(lr=0.01)',
              'RMSprop(lr=0.01)']

for opt_name in optimizers:

    K.clear_session()
    
    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=eval(opt_name),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
    
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([optimizers, metrics_reported],
                                 names=['optimizers', 'metric'])
historydf.columns = idx

ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()    



# ========================================================
#    Initializer for the (random) weights of Keras layers
#       and the bias vector
# ========================================================
dflist = []

initializers = ['zeros', 'uniform', 'normal',
                'he_normal', 'lecun_uniform']

for init in initializers:

    K.clear_session()

    model = Sequential()
    model.add(Dense(1, input_shape=(4,),
                    kernel_initializer=init,
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    h = model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))

# multi-index
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([initializers, metrics_reported],
                                 names=['initializers', 'metric'])

historydf.columns = idx

ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()


# ==============================
#  Inner layer representation
# ==============================
K.clear_session()

model = Sequential()
model.add(Dense(2, input_shape=(4,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

h = model.fit(X_train, y_train, batch_size=16, epochs=20,
              verbose=1, validation_split=0.3)
result = model.evaluate(X_test, y_test)


result
model.summary()

model.layers

inp = model.layers[0].input    # get the first layer's inputs
out = model.layers[0].output   # get the first layer's outputs

inp  # These inp and out are tensors with sizes N*4 and N*2
out             
     
# Define feature function
features_function = K.function([inp], [out])
features_function

features_function([X_test])[0].shape
                 
# this is the output matrix of N*2 of the first layer                 
features = features_function([X_test])[0]    

# visualize after transforming the original 4 features into 2 features, how the data 
# are distributed according to the 2 features.
plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')




# Build a deeper model and try it again!
K.clear_session()

model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

inp = model.layers[0].input
out = model.layers[1].output
features_function = K.function([inp], [out])

plt.figure(figsize=(15,10))

for i in range(1, 26): # iterate over number of epochs
    plt.subplot(5, 5, i)
    h = model.fit(X_train, y_train, batch_size=16, epochs=1, verbose=0) # each time epochs=1. This will build on the previous epochs.
    test_accuracy = model.evaluate(X_test, y_test)[1]
    features = features_function([X_test])[0]
    plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 4.0)
    plt.title('Epoch: {}, Test Acc: {:3.1f} %'.format(i, test_accuracy * 100.0))

plt.tight_layout()

                  