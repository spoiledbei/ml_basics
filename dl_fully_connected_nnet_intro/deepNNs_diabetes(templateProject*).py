# ================================================================
# Created on Thu Dec 20 2017
#
# @author: congshanzhang
# Deep Neural Network
# Binary Classification
# 
# Pima Indians datase with features:
#       Pregnancies: Number of times pregnant
#       Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#       BloodPressure: Diastolic blood pressure (mm Hg)
#       SkinThickness: Triceps skin fold thickness (mm)
#       Insulin: 2-Hour serum insulin (mu U/ml)
#       BMI: Body mass index (weight in kg/(height in m)^2)
#       DiabetesPedigreeFunction: Diabetes pedigree function
#       Age: Age (years)
#
# The goal is to classify artificial 0-1 data
# ================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
step 0) load data
"""
directory = '/Users/congshanzhang/Documents/Office/machine learning/dl_fully_connected_nnet_intro/Data/'
df = pd.read_csv(directory+'diabetes.csv')
df.head()
df.info()
df.describe()


"""
step 1) Visualize Data
"""
# plot the histograms for each feature (Method 1)
plt.figure(figsize=(15, 5))
for i, feature in enumerate(df.columns[:-1]): # except the last column
    plt.subplot(2, 4, i+1)
    df[feature].plot(kind='hist', title=feature)
    plt.xlabel(feature)
    
# plot the histograms for each feature (Method 2)
_ = df.hist(figsize=(12,10))    

# For classification problems, explore correlations of features with labels
import seaborn as sns
sns.pairplot(df, hue="Outcome")

# correlations (heatmap)
sns.heatmap(df.corr(),annot = True)

"""
step 2) Rescale features and prepare X, y
"""
df['Glucose'] = df['Glucose']/100
df['BloodPressure'] = df['BloodPressure']/10
df['SkinThickness'] = df['SkinThickness']/10
df['Insulin'] = df['Insulin']/100
df['BMI'] = df['BMI']/10
df['Age'] = df['Age']/10

"""
# Minmax normalization
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
# To scale the values into the range [0.0, 1.0]
# For each feature, x_scaled = (x - min)/(max-min)
X = mms.fit_transform(df.drop('Outcome', axis=1))
"""

"""
# standard normalization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(df.drop('Outcome', axis=1))
"""

X = df.iloc[:,:-1].values
y = df['Outcome']
y_cat = pd.get_dummies(y).values # change y into categorical data

X.shape # see how many features
y_cat.shape

"""
step 3) train/test split
"""
from sklearn.model_selection import train_test_split
# use 20% testing set. Set seed to be 22
X_train, X_test, y_train, y_test = train_test_split(X, y_cat,test_size=0.2,random_state=22)


"""
step 4) build fully connected NN model
"""
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

K.clear_session() # clear model from memory

# build a 2-layer NN model
model = Sequential()
model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# ============================================================================
# Note: if the output y is not categorical, just a single 0,1 variable.
# Then, we could use the following code for a last layer:
#   model.add(Dense(1, activation='sigmoid'))
#   model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
# ============================================================================

# train model
model.fit(X_train, y_train, epochs=20, verbose = 2, validation_split=0.1)

# test model
# predict probability for each category
y_pred = model.predict(X_test)

# choose the largest probability
y_pred_class = np.argmax(y_pred, axis = 1)
y_test_class = np.argmax(y_test, axis = 1)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("The Accuracy score on the Test set is:\t{:0.3f}".format(accuracy_score(y_test_class, y_pred_class)))
print(classification_report(y_test_class, y_pred_class))

# confusion matrix
def pretty_confusion_matrix(y_true, y_pred, labels=["False(0)", "True(1)"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted '+ l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df

# check the confusion matrix, precision and recall
pretty_confusion_matrix(y_test_class, y_pred_class, labels=['No_Diabetes', 'Diabetes'])

"""
step 5) benchmark
"""
pd.Series(y_test_class).value_counts() / len(y_test_class)







