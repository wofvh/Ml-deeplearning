#pytorch 로 11,12,13 의 예측값을 구하시오

from pickletools import pyint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch :',torch.__version__,'\n','사용DEVICE : ',DEVICE)
print(torch.cuda.device_count())

#1.데이터
#x = np.array([1,2,3,4,5,6,7,8,9,10])
#y = np.array([1,2,3,4,5,6,7,8,9,10])
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([8,9,10])

y_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_test = np.array([8,9,10])

########## 11,12,13 예측값 구하기!########

x_predict = np.array([11,12,13])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)   # (3,) => # (3, 1)
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)  # (3,) => # (3, 1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)  # (3,) => # (3, 1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)  # (3,) => # (3, 1)
x_predict = torch.FloatTensor(x_predict).unsqueeze(-1).to(DEVICE)  # (3,) => # (3, 1)

# print(x_train.shape,x_test.shape)  #(10,1) (3,1)
# print(y_train.shape,y_test.shape)  #(10,1) (3,1)

#standard scaler
print('x_trian:',x_train)  
print('x_test:',x_test) 
print('x_predict:',x_predict)

x_test = (x_test - torch.mean(x_test))/ torch.std(x_test) # Standard Scaler
x_train = (x_train - torch.mean(x_train))/ torch.std(x_train) # Standard Scaler

x_predict = (x_predict - torch.mean(x_predict))/ torch.std(x_predict) # Standard Scaler
print('########################scaler 후##################')

print('x_trian:',x_train)  
print('x_test:',x_test) 
print('x_predict:',x_predict)



# 2.모델구성

model = nn.Sequential(
    nn.Linear(1, 4),
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 3),
    ).to(DEVICE)