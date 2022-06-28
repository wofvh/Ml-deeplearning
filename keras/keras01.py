#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(5))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam') #(mse 평균) (loss 오차)compile (Optimizer(최적화)
model.fit(x, y, epochs=167)

#4. 평가 , 예측
loss = model.evaluate(x, y) #평가값을 로스에 넣는다 #(evaliate 평가)
print('loss : ',loss)

result = model.predict([4])
print('4의 예측값 : ',result )

# loss :  4.721324220469114e-09
# 4의 예측값 :  [[3.9998505]]
