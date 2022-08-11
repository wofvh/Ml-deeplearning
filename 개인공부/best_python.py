from tracemalloc import start
from turtle import shape
from unittest import result
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,fetch_covtype,load_digits
from sklearn.datasets import load_breast_cancer,load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
print('xgboost:', xg.__version__)

datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape)  #(150, 4)  
print(y.shape)

le = LabelEncoder()
y = le.fit_transform(y)
print(y.shape)
