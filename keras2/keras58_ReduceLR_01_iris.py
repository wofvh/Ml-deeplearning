from gc import callbacks
import numpy as np
from torch import dropout
from tensorflow.keras.datasets import cifar100
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D,Conv1D, Flatten, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
import time
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 1. data
#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
print(x_train.shape)#(120, 4)
print(x_test.shape) #(30, 4)

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)


print(x_train.shape)#(120, 4, 1)
print(x_test.shape) #(30, 4, 1)

# 2. model

activation = 'relu'
drop = 0.2

inputs = Input(shape=(4,1), name='input')
x = Conv1D(128, kernel_size=(2,2), activation=activation, padding='valid', name='hidden1')(inputs)
x = Dropout(drop)(x)

x = MaxPooling2D()(x)
x = Conv1D(32, kernel_size=(3, 3), activation=activation, padding='valid', name='hidden3')(x)
x = Dropout(drop)(x) 

# x = Flatten()(x) # (25*25*32) / Flatten의 문제점: 연산량이 너무 많아짐
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Dense(128, activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# 3. compilel, fit
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) # learning rate를 0.5만큼 감축시키겠다

start = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])
end = time.time()

loss, acc = model.evaluate(x_test,y_test)

print('걸린 시간: ', end)
print('loss: ', loss)
print('acc: ', acc)
