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

# Importing the Keras libraries and package

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Initialising the RNN
classifier = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
classifier.add(LSTM(units=50 ,return_sequences= True  , input_shape = (X_train.shape[1] ,1)))
classifier.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units=50 ,return_sequences= True ))
classifier.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
classifier.add(LSTM(units=50 ,return_sequences= True ))
classifier.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation

classifier.add(LSTM(units=50 ))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units=1))

# Compiling the RNN
classifier.compile(optimizer='adam' , loss='mean_squared_error')

# Fitting the RNN to the Training set
classifier.fit(X_train , y_train , epochs = 100 , batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017

# Getting the predicted stock price of 2017


# Visualising the results

