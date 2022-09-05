import numpy as np 

x= np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,11,9,7])

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

learning_rate = 0.1

optimizerlist = [adam, adadelta, adagrad, adamax, rmsprop, nadam]

optimizer = adam.Adam(learning_rate = learning_rate),
optimizer = adadelta.Adadelta(learning_rate = learning_rate),
optimizer = adagrad.Adagrad(learning_rate = learning_rate),
optimizer = adamax.Adamax(learning_rate = learning_rate),
optimizer = rmsprop.RMSprop(learning_rate = learning_rate),
optimizer = nadam.Nadam(learning_rate = learning_rate,)

for i in optimizerlist:
    model.compile(loss='mse', optimizer=i, metrics=['mse'])
    model.fit(x, y, epochs=100, batch_size=1, verbose=0)
    loss, mse = model.evaluate(x, y, batch_size=1)


model.compile(loss='mse', optimizer=optimizer)

model.fit(x,y, epochs=50, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = model.predict([11])
print('loss:',round(loss,4), 'lr : ', learning_rate,'결과물:',y_predict)