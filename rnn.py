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
from sklearn.preprocessing import MinMaxScalar
sc = MinMaxScalar(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output


# Reshaping

# Part 2 - Building the RNN

# Importing the Keras libraries and packages


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

