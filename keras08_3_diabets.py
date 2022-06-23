from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_diabetes

#데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=72)

# print(x)
# print(y)
# print(x.shape, y.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)


#1.모델
model = Sequential()
model.add(Dense(6, input_dim=10))
model.add(Dense(89))
model.add(Dense(225))
model.add(Dense(225))
model.add(Dense(285))
model.add(Dense(155))
model.add(Dense(448))
model.add(Dense(92))
model.add(Dense(1))


#2.컴파일 훈련model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=380, batch_size=18)

#3.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


#결과 loss :  2225.1044921875
#r2스코어 : 0.6285839988523811
