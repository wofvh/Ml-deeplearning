# trainable  = True / False 비교하면서 만들어서 결과비교
from unittest import result
import numpy as np
from keras.models import Sequential
from keras.datasets import cifar100,cifar10
from keras.layers import Dense ,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.applications import VGG16
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
import time
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_test.shape, x_train.shape) # (10000, 32, 32, 3) (50000, 32, 32, 3)
print(y_test.shape, y_train.shape) # (10000, 1) (50000, 1)


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


# model = VGG16() # vgg16은 이미지 분류 모델 #include_top = False : 이미지 분류 모델에서는 마지막 레이어를 제외하고 사용 # if include_top = True , input_shape = (224, 224, 3) 
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) # include_top=False : 이미지 분류 모델에서는 1000개의 노드가 있지만 우리는 10개의 노드만 필요하므로 False로 설정

# vgg16.summary()
# vgg16.trainable=False # 가중치를 동결한다. # vgg16.trainable=True # vgg16을 훈련시킨다. 
# vgg16.summary()

model = Sequential()
model.add(vgg16) 
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(100))

model.trainable = False
                                #trainable:True /vgg16.trainable=False /# model False
model.summary()
print(len(model.weights))           #30 / 30 /30
print(len(model.trainable_weights)) #30 / 4  / 0   # model False          

#####################2번 소스에서 아래만 추가######################################

print(model.layers)

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) # learning rate를 0.5만큼 감축시키겠다

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])
end = time.time()

loss, acc = model.evaluate(x_test,y_test)

print('걸린 시간: ', end)
print('loss: ', loss)
print('acc: ', acc)
