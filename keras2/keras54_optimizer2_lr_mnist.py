import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(123)
#즉시실행모드!!
tf.compat.v1.disable_eager_execution() #

#1. 데이터
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train  = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련

import tensorflow as tf
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

model.fit(x,y, epochs=50, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = model.predict([11])
print('loss:',round(loss,4), 'lr : ', learning_rate,'결과물:',y_predict)
