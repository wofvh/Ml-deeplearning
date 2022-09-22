from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torch 
import torchvision.transforms as tr

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

# # transf = tr.Compose()
# transf = tr.Compose([tr.Resize(120),tr.ToTensor()]) #평균과 표준편차를 정규화한다

path = './_data/torch_data/'

# train_dataset = CIFAR10(path, train=True, download=True,transform=transf)  #train=True 학습용 데이터 #download=True 데이터를 다운로드 받겠다
# test_dataset = CIFAR10(path, train=False, download=True,transform=transf)  #train=False 테스트용 데이터 
train_dataset = CIFAR10(path, train=True, download=True)
test_dataset = CIFAR10(path, train=False, download=True)

x_train , y_train = train_dataset.data/255., train_dataset.targets #데이터를 255로 나누어서 0~1사이의 값으로 만들어준다
x_test , y_test = test_dataset.data/255., test_dataset.targets 

print(x_train.shape ,x_test.size())  #torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
print(y_train.shape ,y_test.size())  #torch.Size([60000]) torch.Size([10000])

print(np.min(x_train.numpy())), np.max((x_train.numpy())) #0.0 1.0
