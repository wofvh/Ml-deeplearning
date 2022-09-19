from cgitb import reset
from tkinter import Widget
import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.impute import KNNImputer
from tqdm.auto import tqdm
import easyocr
import cv2
import matplotlib.pyplot as plt
import wget

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


reader = easyocr.Reader(['en','ko','jp'], gpu=False)
# wget.download('https://meeco.kr/files/attach/images/24268070/716/533/032/17a03beb045782788810949204bd20c1.jpg')

# img_path = ('17a03beb045782788810949204bd20c1.jpg')
# img = cv2.imread(img_path)


# result = reader.readtext(img_path)
# print(result[1])

# THRESHOLD = 0.5


# for bbox, text, conf in result:
#     if conf > THRESHOLD:
#         print(text)
#         cv2.rectangle(img, pt1=bbox[0], pt2=bbox[2], color=(0,255,0), thickness=2)
        
# plt.figure(figsize=(10,10))
# plt.imshow(img[:,:,::-1])
# plt.axis('off')
# plt.show()
