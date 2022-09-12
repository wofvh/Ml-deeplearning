import numpy as np, time
from keras.datasets import cifar100
from keras.applications import EfficientNetB0
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from keras.optimizers import RMSprop, SGD, Nadam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3) (10000, 1)
# print(np.unique(y_train, return_counts=True))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape)   # (50000, 100)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape, y_train.shape)  # (40000, 32, 32, 3) (40000, 100)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 100)
'''
scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
# scaler = MaxAbsScaler()

n = x_train.shape[0]
x_train_reshape = x_train.reshape(n,-1) 
x_train_transform = scaler.fit_transform(x_train_reshape)
x_train = x_train_transform.reshape(x_train.shape) 

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

#x_test_transform = scaler.transform(x_test.reshape(m,-1))
#x_test = x_test_transform.reshape(x_test.shape)
'''
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#2.모델구성
EfficientNetB0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))

model = Sequential()
model.add(EfficientNetB0)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

# es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)  #-> 5번 만에 갱신이 안된다면 (factor=0.5) LR을 50%로 줄이겠다

start = time.time()
model.fit(x_train, y_train, epochs=1000
          
          , batch_size=100, verbose=1, validation_split=0.2, callbacks= [reduce_lr]) 
end = time.time() - start


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))

# learning_rate :  0.001
# loss :  8.6017
# accuracy :  0.0929
# 걸린시간 : 314.4014

# learning_rate :  0.0001
# loss :  3.4049
# accuracy :  0.4083
# 걸린시간 : 1042.1475
# ==================================== preprocess_input(x) ===================================
# loss :  2.7191
# accuracy :  0.5643
# 걸린시간 : 381.4593