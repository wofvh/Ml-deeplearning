#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/gulim.TTc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
from sklearn.datasets import load_boston
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=13))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(9, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

# loss :  30.97882080078125
# r2스코어 :  0.6233822491792897
##################val전후#################
# loss :  17.171226501464844
# r2스코어 :  0.7912448513467571
##################EarlyStopping전후#################
# loss :  16.585956573486328
# r2스코어 :  0.8015628803037077
##################activation전후#################
# loss :  15.895992279052734
# r2스코어 :  0.8098177492286052



# print('==========================')
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001388A0478B0>
# print('==========================')
# print(hist.history)
# print('==========================')
# print(hist.history['loss'])
# print('==========================')
# print(hist.history['val_loss'])


# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', label='loss', color='red')
# plt.plot(hist.history['val_loss'], marker='.', label='val_loss', color='blue')
# plt.grid()
# plt.title('보스턴')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()