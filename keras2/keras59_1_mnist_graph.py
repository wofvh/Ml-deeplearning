import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.layers import GlobalAveragePooling2D
import keras
import time
from gc import callbacks
import numpy as np
from torch import dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2. model

drop=0.2 #여기다가 ,를 찍어서 에러가 났었다. 개빡침

optimizer='adam'
activation='relu'

inputs = Input(shape=(32,32,3), name='input')
x = Conv2D(128,(2,2), activation=activation, padding='valid',  name='hidden1')(inputs) #27.27.128
x = Dropout(drop)(x)

x = MaxPooling2D()(x)
x = Conv2D(32,(3,3),activation=activation, padding='valid', name='hidden3')(x)      #27.27.128
x = Dropout(drop)(x)

# x = Flatten()(x) # (25*25*32) / Flatten의 문제점: 연산량이 너무 많아짐
x = GlobalAveragePooling2D()(x)     #25*25*32 =20000

x = Dense(256, activation=activation,name ="hidden4" )(x)      #27.27.128
x = Dropout(drop)(x)

x = Dense(128, activation=activation,name ="hidden5" )(x)      #27.27.128
x = Dropout(drop)(x)

outputs = Dense(100, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

from keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=20, mode ='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,  mode = 'auto',verbose=1,factor=0.5)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc']) #sparse_categorical_crossentropy

import time 
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr])
end = time.time() - start

loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("learning_rate : ",learning_rate)
print("loss : ", round(loss,4))
print("acc : ", round(acc,4))
print("걸린시간 : ", round(end,4))

##################시각화####################################
import matplotlib.pyplot as plt

plt.figure(figsize=(9,5)) #가로,세로

plt.subplot(2,1,1) #2행 1열중 첫번째
plt.plot(hist.history["loss"], marker='.', c='red', label='loss')
plt.plot(hist.history["val_loss"], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2
plt.subplot(2,1,2) #2행 2열중 두번째
plt.plot(hist.history["acc"], marker='.', c='red',label='a')
plt.plot(hist.history["val_acc"], marker='.', c='blue',label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()