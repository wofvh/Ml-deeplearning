#vgg16 torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from PIL import Image
from torchsummary import summary
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR 