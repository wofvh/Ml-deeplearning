from logging import critical
from pickletools import optimize
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

path = './_data/torch_data/'

