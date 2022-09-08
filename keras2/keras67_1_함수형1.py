from keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras.layers import Dense, Input ,Flatten
from keras.applications import VGG16
from keras.datasets import cifar100
import numpy as np
#함수형으로 만들기 

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_test.shape, x_train.shape) # (10000, 32, 32, 3) (50000, 32, 32, 3)
print(y_test.shape, y_train.shape) #  (10000, 1) (50000, 1)


print(np.unique(y_train, return_counts=True)) 

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler= StandardScaler()              
x_train= x_train.reshape(50000,-1)   
x_test = x_test.reshape(10000,-1)    
                                   
x_train=scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)

x_train= x_train.reshape(50000,32,32,3)
x_test= x_test.reshape(10000,32,32,3)



#2.모델구성
input = Input(shape=(32,32,3))
vgg16 = VGG16(include_top=False)(input)
hidden1 = Dense(100)(vgg16)
output1 = Dense(100, activation='softmax')(hidden1)

model = Model(inputs=input, outputs=output1)

model.summary()

#3. 컴파일, 훈련
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

import time
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5)  #-> 5번 만에 갱신이 안된다면 (factor=0.5) LR을 50%로 줄이겠다

start = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(end,4))