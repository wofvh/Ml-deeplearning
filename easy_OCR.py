from cgitb import reset
from email.mime import image
from tkinter import Widget
from unittest import result
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


THRESHOLD = 0.5
reader = easyocr.Reader(['ja','en',], gpu=True)
# wget.download('https://img.etnews.com/photonews/1909/1227580_20190925141436_569_0003.jpg')

img_path = '1227580_20190925141436_569_0003.jpg'

# result = reader.readtext(img_path, detail=0)
def read(img_path):
    img = cv2.imread(img_path)
    
    result = reader.readtext (img_path)  
    r = []


    for bbox, text, conf in result:
        if conf > THRESHOLD:
            print(text)
    cv2.rectangle(img, pt1=bbox[0][0], pt2=bbox[2], color=(0,255,0), thickness=2)
    plt.figure(figsize=(10,10))
    plt.imshow(img[:,:,::-1])
    plt.axis('off')

plt.show()

    
