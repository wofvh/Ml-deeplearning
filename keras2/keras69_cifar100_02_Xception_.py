from re import X
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications import Xception
import time 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
#1. 데이터
(x_train , y_train), (x_test,y_test) = cifar100.load_data()

print(x_test.shape,x_train.shape)
print(y_test.shape,y_train.shape)

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    train_size=0.8, shuffle=True, random_state=66)

xception = Xception(weights='imagenet', include_top=False, input_shape=(32,32,3))

xception.trainable = False 

# scaler = MaxAbsScaler()
# n = x_train.shape[0]
# x_train_reshape = x_train.reshape(n,-1) 
# x_train_transform = scaler.fit_transform(x_train_reshape)
# x_train = x_train_transform.reshape(x_train.shape) 


model = Sequential()
model.add(xception)
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(100, activation = 'softmax'))
model.summary()


'''
from keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr  = learning_rate)

model.compile(loss ='categorical_crossentropy', optimizer = optimizer, metrics=["accuracy"])

es = EarlyStopping(moniter = 'val_loss', patience=5, mode='auto',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience= 5,mode = 'auto', verbose=1, factor=0.5)


start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.25, callbacks=[es, reduce_lr]) 
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("learning_rate: ", learning_rate)
print('loss: ', round(loss,4))
print('accuracy: ', round(accuracy,4))
print("걸린시간: ", round(end - start,4))
'''