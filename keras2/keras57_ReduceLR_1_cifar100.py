import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
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

from tensorflow.keras.utils import to_categorical
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

model.compile(optimizer=optimizer, metrics = ['acc'],
                loss='categorical_crossentropy')

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=20, mode ='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,  mode = 'auto',verbose=1,factor=0.5)
start = time.time()
model.fit(x_train,y_train, epochs=10 , validation_split=0.4, callbacks=[es,reduce_lr],batch_size=128)#2개이상은 리스트[]로 묶어줘야함
end = time.time()

loss, acc = model.evaluate(x_test,y_test)

print('걸린시간 : ', end )
print("loss : ", loss)
print('acc : ', acc)

# print('acc : ', accuracy_score(y_test,y_predict))

# 걸린 시간:  392.6866011619568
# loss:  2.7481682300567627
# acc:  0.3084000051021576