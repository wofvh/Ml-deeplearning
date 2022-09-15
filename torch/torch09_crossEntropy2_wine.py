#logistic_regression 회기모델 (시그모이드 함수)2 진분류 0 N 1 

from calendar import EPOCH
from tkinter import Y
from unittest import result
from sklearn.datasets import load_wine

import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


datasets = load_wine()
x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.2, random_state=42 ,) #stratify=y)

print(x_train.size())
print(x_test.)
print(y_train)
print(y_train)


x_train = torch.FloatTensor(x_train)
