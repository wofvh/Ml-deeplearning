from sklearn import datasets
import tensorflow as tf
from mimetypes import init
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./_data/project/cls_data.csv',index_col=0)

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):  
        subset = dataset[i : (i + size)]
        aaa.append(subset)       
    return np.array(aaa)

bbb = split_x(dataset,size)
print(bbb.shape) 


x = bbb[:, :-1]
y = bbb[:,  -1]
print(x)
print(y)

print(x.shape,y.shape)   #x(4365, 4, 2) # y(4365, 2)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=31
                                                    )

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


'''

    
def get_data(data):
    batch_x = []
    batch_y = []
    
    for _ , r in data.iterrows(): # data iterrows 로 한개씩 훑고 지나감(이미지를 불러오기 위함) 연습필요 당장 이해 힘듬
        file_path = r['image_path'] 
        
        image = plt.imread(file_path) #이미지 불러오고
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cvtColor 할거임

        image = cv2.resize(image , (self.image_size , self.image_size)) # model input size로 resize 해주기
        image = image / 255. # 정규화 0~1

        # label = r.iloc[:,2:-1] # label로 지정된 int형 (정답지 : ground Truth)
        label = r['label']
        batch_x.append(image) # 앞선 리스트에 정의
        batch_y.append(label)

    return batch_x, batch_y # return => __getitem__
'''