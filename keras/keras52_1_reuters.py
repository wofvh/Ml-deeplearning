from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train,y_train),(x_test,y_test)= reuters.load_data(
    num_words=100, test_split=0.2,
)


print(x_train)
print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train)
print(np.unique(y_train, return_counts = True))      #46개의 뉴스카테고리
print(len(np.unique(y_train)))                       #46

print(type(x_train), type(x_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))             #<class 'list'> 일정하지 않음
print(len (x_train[0]))             #56  (8982,)<<전부다 다른길이 
print(len (x_train[1]))             #71

print(len(max(x_train)))            #83
          
print("뉴스기사의 최대길이 :", max(len(i) for i in x_train)) #2376
print("뉴스기사의 평균길이 :", sum(map(len,x_train)) / len(x_train)) #뉴스기사의 평균길이 : 145.5398574927633\

#전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre',maxlen=100,truncating='pre') #(8928,) > (8928)


x_test = pad_sequences(x_test, padding='pre',maxlen=100,truncating='pre') #(8928,) > (8928)

y_train = to_categorical(y_train)
y_train = to_categorical(y_test)

print(x_train.shape,y_train.shape)  #(8982, 100) (2246, 46)
print(x_test.shape,y_test.shape)    #(2246, 100) (2246,)

#2.모델 구성
