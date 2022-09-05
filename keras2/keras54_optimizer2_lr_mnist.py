import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D 
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.datasets import mnist

tf.compat.v1.set_random_seed(123)

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train  = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.


print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape) 

'''
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2),   # 출력(4,4,10)                                       # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                 padding='same',
                 input_shape=(None,28, 28, 1)))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라 
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


from tensorflow.python.keras.optimizer_v2 import adam , adadelta , adagrad , adamax 
from tensorflow.python.keras.optimizer_v2 import rmsprop ,nadam

learning_rate = 0.0010

# optimizer = adam.Adam(learning_rate = learning_rate)
# optimizer = adadelta.Adadelta(learning_rate = learning_rate)
# optimizer = adagrad.Adagrad(learning_rate = learning_rate)
# optimizer = adamax.Adamax(learning_rate = learning_rate)
# optimizer = rmsprop.RMSprop(learning_rate = learning_rate)
optimizer = nadam.Nadam(learning_rate = learning_rate)

model.compile(loss='mse', optimizer=optimizer)
#4. 평가, 예측\
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)
'''

