#logistic_regression 회기모델 (시그모이드 함수)2 진분류 0 N 1 

from calendar import EPOCH
from tkinter import Y
from sklearn.datasets import load_breast_cancer

import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


datasets = load_breast_cancer()
x = datasets.data 
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.2, random_state=42 , stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)


print('x_trian:',x_train)  
print('x_test:',x_test) 


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('########################scaler 후##################')

print('x_trian:',x_train)  
print('x_test:',x_test) 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size()) #([113, 30]
print(x_train.shape)  #[113, 30]

#2. 모델구성
model  = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss().to(DEVICE) #바이너리 크로스 엔트로피 BCE #  criterion 표준,기준

optimizer  = optim.Adam(model.parameters(), lr=0.01) # model.parameters() 모델의 가중치를 가져옴 #adam 옵티마이저 #lr 학습률


def train(model, criterion , optimizer , x_train, y_train):
    model.train() # 훈련모드로 바꿔줌 써도되고 안 써도됨 ^^
    optimizer.zero_grad()#잔여 미분값 초기화 #필수정의
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train) # criterion 표준,기준
    # y_pred = model(x_train) #모델에 x_train을 넣어서 y_pred를 예측
    # loss = criterion(y_pred, y_train) #예측값과 y_train을 비교해서 loss를 구함
    loss.backward() # 역전파를 실행하게됨 ! #필수정의
    optimizer.step()# 가중치를 갱신한다 
    return loss.item() #loss.item() 스칼라값을 반환 

EPOCHS = 100
for epoch in range(1,EPOCHS + 1):   
    loss = train(model, criterion , optimizer , x_train, y_train)
    print('epoch {}, loss: {:.8f}'.format(epoch, loss)) 


#4. 평가, 예측
print('======================평가, 예측======================')




