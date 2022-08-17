import pandas as pd
import random
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier,XGBRegressor  
import matplotlib.pyplot as plt

path = 'D:/study_data/dacon/'

data = pd.read_csv(path + 'fish.csv',index_col=0)

print(data)  #[159 rows x 6 columns]
print(data.info())

print(data.isna().sum())

    