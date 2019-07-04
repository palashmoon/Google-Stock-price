# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[: , 1:2].values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scale = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train=[]
y_train=[]
for i in range(60 , 1258):
    X_train.append(training_set_scale[i-60:i , 0])
    y_train.append(training_set_scale[i , 0])

X_train , y_train = np.array(X_train) , np.array(y_train)


# Reshaping
X_train = np.reshape(X_train , (X_train.shape[0] , X_train.shape[1] , 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages

from keras.model import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Initialising the RNN

# Adding the first LSTM layer and some Dropout regularisation


# Adding a second LSTM layer and some Dropout regularisation


# Adding a third LSTM layer and some Dropout regularisation

# Adding a fourth LSTM layer and some Dropout regularisation


# Adding the output layer

# Compiling the RNN


# Fitting the RNN to the Training set



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017

# Getting the predicted stock price of 2017


# Visualising the results

