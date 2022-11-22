import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential() 
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights) # 가중치
print('====================================================================')
print(model.trainable_weights) # 훈련가능한 가중치
print(len(model.weights))
print(len(model.trainable_weights))

model.trainable = False # 훈련을 시키지 않겠다.

print('====================================================================')

print(len(model.weights))
print(len(model.trainable_weights))

print('====================================================================')
print(model.trainable_weights) # 가중치

model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1)

y_predict = model.predict(x)
print(y_predict[:3])
# loss = model.evaluate(x, y)
# print('loss : ', loss)