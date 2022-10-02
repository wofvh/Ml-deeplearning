import torch
print(torch.__version__)
import torchvision.transforms as tr
##################### Transform, Data Set 준비
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torch 


USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

path = './_data/torch_data/'

transf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# trainset = torchvision.datasets.STL10(root='./_data/torch_data/', split='train', download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# testset = torchvision.datasets.STL10(root='./_data/torch_data/', split='test', download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

train_dataset = torchvision.datasets.CIFAR10(path, train=True, download=False)  #train=True 학습용 데이터 #download=True 데이터를 다운로드 받겠다
test_dataset = torchvision.datasets.CIFAR10(path, train=False, download=False)  #train=False 테스트용 데이터 
# train_dataset = CIFAR10(path, train=True, download=True)
# test_dataset = CIFAR10(path, train=False, download=True)

x_train , y_train = train_dataset.data/255., train_dataset.targets #데이터를 255로 나누어서 0~1사이의 값으로 만들어준다
x_test , y_test = test_dataset.data/255., test_dataset.targets 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape ,x_test.size()) #torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])
print(y_train.shape ,y_test.size()) #torch.Size([50000]) torch.Size([10000])
x_train, x_test = x_train.reshape(50000,3,32,32),x_test.reshape(10000,3,32,32)

print(x_train.shape,x_test.shape) 
# exit()
train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)


train_loader = DataLoader(train_dset, batch_size=32, shuffle =True)#batch_size=32 한번에 32개씩 불러온다 #shuffle=True 데이터를 섞어준다
test_loader =  DataLoader(test_dset , batch_size=32, shuffle =False)

print(len(train_loader), len(test_loader)) #1563 313



plt.imshow(train_loader[0].permute(1,2,0)).to(DEVICE)
