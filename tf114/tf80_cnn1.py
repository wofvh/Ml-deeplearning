from sys import float_repr_style
import tensorflow as tf
import keras
import numpy as np

tf.compat.v1.set_random_seed(123)

#1. 데이터
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train  = to_categorical(y_train)
x_test = to_categorical(x_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.


#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None,28,28,1]) #imput_shape
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10]) #output_shape

#3. 컴파일, 훈련
w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,64])#filter #1은 흑백, 3은 컬러 #64는 노드의 갯수
                                        #커널싸이즈, 컬러, 필터

L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1], padding='SAME') #SAME은 패딩을 해준다. #VALID는 패딩을 안해준다.
                        