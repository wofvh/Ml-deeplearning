from logging import critical
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import torchvision.transforms as tr
from torchsummary import summary
USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


path = './_data/torch_data/'

transf = tr.Compose([tr.Resize(224,224),tr.ToTensor(),tr.Normalize])

# train_dataset = CIFAR10(path, train=True, download=True)  #train=True 학습용 데이터 #download=True 데이터를 다운로드 받겠다
# test_dataset = CIFAR10(path, train=False, download=True)  #train=False 테스트용 데이터 
train_dataset = CIFAR10(path, train=True, download=True)
test_dataset = CIFAR10(path, train=False, download=True)

x_train , y_train = train_dataset.data/255., train_dataset.targets #데이터를 255로 나누어서 0~1사이의 값으로 만들어준다
x_test , y_test = test_dataset.data/255., test_dataset.targets 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape ,x_test.size())  
print(y_train.shape ,y_test.size())
# print(x_test,y_test)

print(np.min(x_train.numpy())), np.max((x_train.numpy())) #0.0 1.0

print(x_train.shape ,x_test.size())  #torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])


x_train , x_test = x_train.reshape(50000,32*32*3), x_test.reshape(10000,32*32*3) #데이터를 3차원으로 만들어준다

print(x_train.shape ,x_test.size())  #torch.Size([50000, 3072]) torch.Size([10000, 3072])

print(y_test.unique())

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)


train_loader = DataLoader(train_dset, batch_size=32, shuffle =True)#batch_size=32 한번에 32개씩 불러온다 #shuffle=True 데이터를 섞어준다
test_loader =  DataLoader(test_dset , batch_size=32, shuffle =False)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,128,3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128,128,3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128,256,3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256,512,3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,3, padding=1),nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.avg_pool = nn.AvgPool2d((7, 7))
        self.Classifier = nn.Linear(512, 1000)
        
        self.avg_pool = nn.AvgPool2d(7)
        #512 1 1
        self.classifier = nn.Linear(512, 1000)
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        """

    def forward(self, x):
        #print(x.size())
        features = self.conv(x)
        #print(features.size())
        x = self.avg_pool(features)
        #print(avg_pool.size())
        x = x.view(features.size(0), -1)
        #print(flatten.size())
        x = self.classifier(x)
        #x = self.softmax(x)
        return x, features
        
        
        
        
model = Net(3).to(DEVICE)
from torchsummary import summary
summary(model, (3, 32,32))#torch summary를 사용하면 모델의 구조를 볼수있다
exit()
