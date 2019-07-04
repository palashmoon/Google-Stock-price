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
 we are interseted only in the open coloumn of the Google_stock_price csv file thats the reason we extracted only open coloumn from our dataset