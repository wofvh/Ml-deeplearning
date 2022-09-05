import tensorflow as tf 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np 
from tensorflow.python.keras.utils import to_categorical
#1데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train , x_test , y_train , y_test = train_test_split(x,y,train_size = 0.9, random_state = 123,
                                                       stratify = y)

print(x.shape, y.shape) #(569, 30) (569, 2)
print(x_train.shape ,y_train.shape) #(512, 30) (512, 2)
print(x_test.shape ,y_test.shape)   #(57, 30) (57, 2)

x = tf.placeholder(tf.float32, shape=[None,x.shape[1]])
y = tf.placeholder(tf.float32, shape=[None,y.shape[1]])


w1 = tf.Variable(tf.random_normal([x.shape[1],10]))
b1 = tf.Variable(tf.random_normal([10]))
h1 = tf.nn.relu(tf.matmul(x,w1)+b1)