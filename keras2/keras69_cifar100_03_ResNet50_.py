from pickletools import optimize
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG19 ,ResNet50
from keras.callbacks import EarlyStopping
import numpy as np
from keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import preprocess_input
from keras.utils import to_categorical
import time
#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar100.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test) #to_categorical : 원핫인코딩

x_train = preprocess_input(x_train) #preprocess_input은 이미지 전처리를 해준다.
x_test = preprocess_input(x_test)

print(x_train.shape,x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)

resnet50  = ResNet50(weights='imagenet',include_top=False,input_shape=(32,32,3)) #resnet50 은 include_top이 False가 기본값

model = Sequential()
model.add(resnet50)
model.add(GlobalAveragePooling2D())
model.add(Dense(100,activation='softmax'))
model.summary()
#3. 컴파일 &훈련
from keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr  = learning_rate)

model.compile(loss = 'categorical_crossentropy', optimizer= optimizer,metrics=['acc'])

es = EarlyStopping(monitor='val_loss',patience=5,mode='auto',verbose=1,restore_best_weights=True)
reduce_lr = RecursionError(monitor='val_loss',patience = 5,mode = 'auto',verbose=1,factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))
