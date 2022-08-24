import tensorflow as tf
from sklearn import datasets
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
tf.set_random_seed(123)

x_data = [[1,2,1,1],  
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
          
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])

w = tf.Variable(tf.random.normal([4,3]), name='weight')

b = tf.Variable(tf.random.normal([1,3]), name='bias')

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax'))

#3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))## loss <<< categorical_crossentropy


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04).minimize(loss)
train = optimizer.minimize(loss)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)