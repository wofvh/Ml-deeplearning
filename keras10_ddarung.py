
import numpy as new_panel
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#1. 데이터

path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
print(train_set)
print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv',index_col=0) #iddex_col 0 번째 위치함
print(test_set)
print(test_set.shape) # (715, 9)

print(train_set.columns)
print(train_set.info())               #unll 중간중간 없는데이터 #결측치 이빨빠진 데이터
print(train_set.describe()) 

### 결측치 처리1. 제거 ###
print(train_set.isnull().sum())
train_set = train_set.dropna()
print(train_set.isnull().sum())

x = train_set.drop(['count'], axis=1) #drop 지울떄 사용함
print(x)
print(x.columns)
print(x.shape)  #(1459, 9)

y = train_set['count']
print(y)
print(y.shape) #(1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=62)


#2. 모델구성
model = Sequential()
model.add(Dense(56, input_dim=9))
model.add(Dense(51))
model.add(Dense(43))
model.add(Dense(44))
model.add(Dense(23))
model.add(Dense(75))
model.add(Dense(42))
model.add(Dense(1))

#3. 컴파일 , 훈련 
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 892, batch_size=52)

#평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)




#y_predict = model.predict(test_set)

#함수 정의 하기

