from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,TensorDataset #데이터를 합쳐주는 역할을한다 
import torch.nn as nn
import torch.optim as optim
import numpy as np

path = './_data/torch_data/'

train_dataset = MNIST(path, train=True, download=True)  #train=True 학습용 데이터 #download=True 데이터를 다운로드 받겠다
test_dataset = MNIST(path, train=False, download=True)  #train=False 테스트용 데이터 

x_train , y_train = train_dataset.data/255., train_dataset.targets #데이터를 255로 나누어서 0~1사이의 값으로 만들어준다
x_test , y_test = test_dataset.data/255., test_dataset.targets 

print(x_train.shape ,x_test.size())  #torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
print(y_train.shape ,y_test.size())  #torch.Size([60000]) torch.Size([10000])


print(np.min(x_train.numpy())), np.max((x_train.numpy())) #0.0 1.0

x_train, x_test = x_train.view(-1, 28*28), x_test.view(-1, 28*28) #28*28 = 784 #view는 차원을 바꿔준다(reshape와 같은 역할)

print(x_train.shape ,x_test.size())  #torch.Size([60000, 784]) torch.Size([10000, 784])