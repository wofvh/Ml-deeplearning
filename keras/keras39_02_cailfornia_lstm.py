from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np
import time
from pathlib import Path
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM

#1. 데이터
datasets = fetch_california_housing()

x, y = datasets.data, datasets.target


print(x.shape,y.shape) #(506, 13) (506,)

# x = x.reshape(506,13,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )


print(x_train.shape, x_test.shape)  # (16512, 2, 2, 2) (4128, 2, 2, 2) 


#2. 모델구성
'''
model = Sequential()
model.add(Conv2D(filters = 200, kernel_size=(3,3), # kernel_size = 이미지 분석을위해 2x2로 잘라서 분석하겠다~
                   padding='same', # padding : 커널 사이즈대로 자르다보면 가생이는 중복되서 분석을 못해주기때문에 행렬을 키워주는것, 패딩을 입혀준다? 이런 너낌
                   input_shape=(2,2,2)))
model.add(Conv2D(filters = 200, kernel_size=(3,3), # kernel_size = 이미지 분석을위해 2x2로 잘라서 분석하겠다~
                   padding='same'))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

model = Sequential()
# model.add(Dense(units=10, input_shape = (3,)))       
# model.summary()
# (input_dim + bias) * units = summary Param # (Dense 모델)
model.add(Dense(64, input_shape = (28*28*1,)))
model.add(Flatten())  # (N, 5408)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict.shape)
print(y_test.shape)
# y_predict = y_predict.reshape(102,13)
# y_test = y_test.reshape(102,)

print(y_test.shape)
print(y_predict.shape)
print(y_test)
print(y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)

# loss :  11.383050918579102
# r2스코어 :  0.8638112999770811

# loss :  0.37719547748565674
# r2스코어 :  0.7294276537739492
'''