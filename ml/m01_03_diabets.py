import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
#1. 데이터
datasets= load_diabetes()
x = datasets['data']
y = datasets['target']

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.9,shuffle=True,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_test.shape)  #(45, 10)

#2. 모델구성
model = LinearSVC()
#3. 컴파일,훈련

model.fit(x_train,y_train)

#4. 평가,예측
results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
#ex)분류는 ACC 회귀는 R2스코어
print("results :",results)