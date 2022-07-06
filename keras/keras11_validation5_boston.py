from tabnanny import verbose
from sqlalchemy import false, true
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score 

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=77)
print(x,y)
'''
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(98))
model.add(Dense(72))
model.add(Dense(85))
model.add(Dense(95))
model.add(Dense(82))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 1000,
           batch_size=35, verbose=1, validation_split = 0.2 )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
'''