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


#                   loss: 6163.0708 lr :  0.1 결과물: [[179.78178]]
# adam.Adam         loss: 1.9133 lr :  0.001 결과물: [[9.889087]]
# adadelta.Adadelta loss: 2.7296 lr :  0.001 결과물: [[9.078296]]
# adagrad.Adagrad   loss: 1.9729 lr :  0.001 결과물: [[9.934455]]
#adamax.Adamax      loss: 2.6313 lr :  0.001 결과물: [[9.046783]]
#rmsprop.RMSprop    loss: 8.671 lr :  0.001 결과물: [[6.063696]]
#nadam.Nadam        loss: 1.7611 lr :  0.001 결과물: [[10.458817]]
