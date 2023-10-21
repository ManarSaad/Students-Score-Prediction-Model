"""

A code to implement a lineara regreission model
to predict student score, depending on number of feature.  Note that I have focused on making the code
simple, easily readable.  It is not including the EDA process, 
"""

#### Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Read the data to pandas dataframe
dataset = pd.read_csv('Student_Performance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Check nulls
dataset.isnull().sum()

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Check the accurcy of our model using two metrics
# 1
def mse(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
   
    # Summing square differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += (prediction - target)**2
       
    # Calculating mean and dividing by 2
    mae_error = (1.0 / (2*samples_num)) * accumulated_error
   
    return mae_error
mse(y_pred,y_test)


# 2
r2_score(y_test, y_pred)  