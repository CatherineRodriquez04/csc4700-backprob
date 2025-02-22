'''
Catherine Rodriguez
Project 1 - CSC 4700
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def dataloader():
    file_path = 'mpg/auto-mpg.data'

    # Load the dataset 
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, na_values=['?'])
    data = data.dropna()
    cleaned_data = data.dropna(subset=[0])

    # Identify numeric columns only for features
    numeric_columns = cleaned_data.select_dtypes(include=['number']).columns

    X = cleaned_data[numeric_columns].drop(0, axis=1)  # Drop the first column 
    y = cleaned_data[0]  # The first column is the target 

    # Ensure y is a 1D numpy array
    y = y.values 

    # Split the data into training, validation, and testing sets
    X_train, X_leftover, y_train, y_leftover = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)  # 70% training, 30% testing & validation
    X_val, X_test, y_val, y_test = train_test_split(X_leftover, y_leftover, test_size=0.5, random_state=42, shuffle=True)  # Split the 30% into 15% for validation and 15% for testing

    # Standardize the data 
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    if np.any(X_std == 0):
        print("Warning: Some features have zero variance!")
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Standardize the target variable 
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        print("Warning: Target variable has zero variance!")
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Return the preprocessed data
    return X_train, X_val, X_test, y_train, y_val, y_test, y_mean, y_std
