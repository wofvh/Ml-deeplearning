from sklearn.datasets import load_digits
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
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)   #(1797, 64) (1797,)   
print(np.unique(y, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                                        #, array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))