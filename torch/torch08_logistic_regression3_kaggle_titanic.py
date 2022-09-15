
from calendar import EPOCH
from tkinter import Y
from unittest import result
from sklearn.datasets import load_digits
import pandas as pd
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path +'train.csv')

test_set = pd.read_csv(path + 'test.csv',index_col=0)  # index_col=n n번째 컬럼을 인덱스로 인식
