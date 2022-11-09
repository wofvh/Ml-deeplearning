import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.impute import KNNImputer
import easyocr
import cv2
import matplotlib.pyplot as plt
import wget
USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

THRESHOLD = 0.5
reader = easyocr.Reader(['ja','en',], gpu=False)
# wget.download('https://img.etnews.com/photonews/1909/1227580_20190925141436_569_0003.jpg')

img_path = '1227580_20190925141436_569_0003.jpg'
img = cv2.imread(img_path)
# result = reader.readtext(img_path, detail=0)
def read(img_path):
    img = cv2.imread(img_path)
    
    result = reader.readtext (img_path)  
    r = []

    for bbox, text, conf in result:
        if conf > THRESHOLD:
            print(text)
    cv2.rectangle(img, pt1=bbox[0][0], pt2=bbox[2], color=(0,255,0), thickness=2)

    print(r)
plt.figure(figsize=(10,10))
plt.imshow(img[:,:,::-1])
plt.axis('off')
plt.show()
read(img_path)
print(img_path)
