print #https://www.kaggle.com/competitions/bike-sharing-demand/discussion?page=7
import numpy as np
import pandas as pd
from pyrsistent import b
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import datetime as dt
#1. 데이터

path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv',)#(10886, 11)

test_set = pd.read_csv(path + 'test.csv',)  #(6493, 8)

######## 년, 월 ,일 ,시간 분리 ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # 트레인 세트에서 캐주얼 레지스터드 드랍
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)
print(test_set)
##########################################

x = train_set.drop(['count'],axis=1) #drop 데이터에서 '' 사이 값 빼기
print(x)
print(x.columns)
print(x.shape)    #(10886, 12)

y = train_set['count']
print(y)
print(y.shape) #(10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=31 )


#2. 모델구성
model = Sequential()
model.add(Dense(100, activation ='swish',input_dim=12))
model.add(Dense(100, activation ='elu'))
model.add(Dense(100, activation ='swish'))
model.add(Dense(100, activation ='elu'))
model.add(Dense(1))

#3. 컴파일 , 훈련 
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =1, batch_size=100, verbose=1 )

#평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape) #(715, 1)

submission_set = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(submission_set)

submission_set['count'] = abs(y_summit)
print(submission_set)

submission_set.to_csv(path + 'submission.csv', index = True)