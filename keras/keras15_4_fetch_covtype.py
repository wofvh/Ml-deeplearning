from sklearn.datasets import fetch_covtype
from unittest import result
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf
tf.random.set_seed(66)

#1.데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)  #(581012, 54) (581012,)
print(np.unique(y, return_counts=True))    #[1 2 3 4 5 6 7]      
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],  
#       dtype=int64))

# print(datasets.DESCR)
# print(datasets.feature_names)
# x = datasets['data']
# y = datasets['target']