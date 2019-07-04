# Google-Stock-price

# Dependencies
``` 
1. pip install numpy
2. pip install panda
3. pip install matplotlib
4. pip install scikit-learn
5. pip install keras 

```

## Import libraries
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Importing training set
```
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[: , 1:2].values
```
 we are interseted only in the open coloumn of the Google_stock_price csv file thats the reason we extracted only open coloumn from our dataset.

 ## Feature Scaling
 ### 1.why feature scaling?
  If a feature in the dataset is big in scale compared to others then in algorithms where Euclidean distance is measured this big scaled feature becomes dominating and needs to be normalized.

### Examples of Algorithms where Feature Scaling matters 
1. K-Means uses the Euclidean distance measure here feature scaling matters.
2. K-Nearest-Neighbours also require feature scaling.
3. Principal Component Analysis (PCA): Tries to get the feature with maximum variance, here too feature scaling is required.
4. Gradient Descent: Calculation speed increase as Theta calculation becomes faster after feature scaling.

Note: Naive Bayes, Linear Discriminant Analysis, and Tree-Based models are not affected by feature scaling.
In Short, any Algorithm which is Not Distance based is Not affected by Feature Scaling.

```
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scale = sc.fit_transform(training_set)
```
We have two choice for feature scaling 1. standard deviation 2. Normalization
here we are using Normalization bcoz Normalization is more suited in RNN and in the activation fuction where Sigmoid function is use at the output.

## Create a data Structure