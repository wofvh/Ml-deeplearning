import tensorflow as tf #2.8.1
print(tf.__version__)
import numpy as np
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D 
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping 

tf.compat.v1.set_random_seed(123)

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

from keras.utils import to_categorical

y_train  = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.


print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape) 

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2),   # 출력(4,4,10)                                       # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                 padding='same',
                 input_shape=(28, 28, 1)))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라 
                                                                                           # kernel_size(2*2) * 바이어스(3) + 10(output)
model.add(MaxPooling2D())

 #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
model.add(Conv2D(32, (2,2), 
                 padding = 'valid',         # 디폴트값(안준것과 같다.) 
                 activation= 'relu'))    # 출력(3,3,7)                                                     
model.add(Flatten()) # (N, 63)
model.add(Dense(16, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()


#3. 컴파일, 훈련

#3. 컴파일, 훈련
from keras.optimizers import Adam,Adadelta,Adagrad,Adamax,RMSprop,SGD,Nadam
from tensorflow.python.keras.optimizer_v2 import adam, adadelta,adagrad,adamax,rmsprop,nadam

learning_rate = 0.001

optimizers = [adam.Adam(lr=learning_rate) ,adadelta.Adadelta(lr=learning_rate ),adagrad.Adagrad(lr=learning_rate ),
              adamax.Adamax(lr=learning_rate) ,rmsprop.RMSprop(lr=learning_rate ) ,nadam.Nadam(lr=learning_rate ) ]
aa = []
for i in optimizers :
    model.compile(loss='categorical_crossentropy',optimizer = i, metrics=['accuracy'])
    earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
              verbose=1, restore_best_weights = True) 
    
    model.fit(x_train,y_train,epochs=2,batch_size=5000,verbose=1, validation_split=0.2,
              callbacks=[earlystopping])
    y_test = tf.argmax(y_test,axis=1) 
    results = model.evaluate(x_test,y_test)
    y_predict = model.predict(x_test)
    y_predict = tf.argmax(y_predict,axis=1) 
    
    
    acc = accuracy_score(y_test,y_predict)
    
    
    print('results:',results,i,'acc:',acc)
    aa.append(acc)
    print(aa)
    

model.compile(loss='sparse_catrgorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
              verbose=1, restore_best_weights = True)     

import time 
start = time.time()
        
hist = model.fit(x_train, y_train, epochs=20, batch_size=3000,verbose=1,
                 validation_split=0.2, callbacks=[earlystopping])
end =  time.time()- start

results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)