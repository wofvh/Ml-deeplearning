from logging import critical
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torch 
import torchvision.transforms as tr
import torchvision.transforms as transforms
USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

path = './_data/torch_data/'

transf = tr.Compose([tr.Resize(15),tr.ToTensor()])

train_dataset = CIFAR10(path, train=True, download=True)  #train=True 학습용 데이터 #download=True 데이터를 다운로드 받겠다
test_dataset = CIFAR10(path, train=False, download=True)  #train=False 테스트용 데이터 
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

# print(np.min(x_train.numpy())), np.max((x_train.numpy())) #0.0 


# x_train , x_test = x_train.unsqueeze(1),x_test.unsqueeze(1) #차원 바꿔주기 

x_train, x_test = x_train.reshape(50000,3,32,32),x_test.reshape(10000,3,32,32)

print(x_train.shape,x_test.shape) 
# exit()
train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)


train_loader = DataLoader(train_dset, batch_size=32, shuffle =True)#batch_size=32 한번에 32개씩 불러온다 #shuffle=True 데이터를 섞어준다
test_loader =  DataLoader(test_dset , batch_size=32, shuffle =False)

print(len(train_loader), len(test_loader)) #1563 313

VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512        
        )
        self.fc_layer = nn.Sequential(
            # CIFAR10은 크기가 32x32이므로 
            nn.Linear(8*base_dim*1*1, 4096),
            # IMAGENET이면 224x224이므로
            # nn.Linear(8*base_dim*7*7, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_layer(x)
        return x