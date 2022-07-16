#LSTM , Conv1D까지  써서  평가 까지 스플릿트 시계열로 자르기 

print #https://www.kaggle.com/competitions/bike-sharing-demand/discussion?page=7
import numpy as np
import pandas as pd
from pyrsistent import b
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import datetime as dt
#1. 데이터

path = './_data/kaggle_jena/'

a = pd.read_csv(path +'jena_climate_2009_2016.csv',index_col=0)

print(a)
print(a.shape)  #(420551, columns 14)

x_predict = np.array(range(96, 106))

size1 = 5                               #<<< 이럴경우 x5 개 y는 1 개

###########시계열 데이터####원하는 y 값을 만들어준다 
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):  
        subset = dataset[i : (i + size)]
        aaa.append(subset)       
    return np.array(aaa)

bbb = split_x(a, size1)
print(bbb.shape)  

x = bbb[:, :-1]
y = bbb[:,  -1:]

# k = ccc[:, -1]
print(x)
print(y.shape) 

# x = x.reshape(96,4,1)
# y = y.reshape(96,1,1)
# print(x_predict.shape)

