import pandas as pd
import random
import os
import numpy as np 

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import _MultiOutputEstimator

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) #seed 고정

train_df = pd.read_csv("./_data/prodacon/")

train_y = train_df.filter(regex ="X")  #input: x featrue
train_x = train_df.filter(regex ="y")  #output: y Feature

