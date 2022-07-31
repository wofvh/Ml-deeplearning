from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import math
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input 

#1. 데이터
path = './_data/project/'
train_set = pd.read_csv(path + 'cls_data.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'cls_data.csv', # 예측에서 쓸거임                
                       index_col=0)

print(test_set)