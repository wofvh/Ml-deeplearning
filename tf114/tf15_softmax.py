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

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 3-2.훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 500
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, hypothesis, train],
                                    feed_dict = {x:x_data, y:y_data})
    if epochs %20 == 0:
        print(epochs, 'loss : ', cost_val, 'hy_val : ', hy_val)

# 4.평가, 예측
y_predict = sess.run(tf.argmax(hy_val,axis=1))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
y_data = sess.run(tf.argmax(y_data,axis=1))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
                                                                       #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_data,y_predict)
print('acc : ', acc)


sess.close()