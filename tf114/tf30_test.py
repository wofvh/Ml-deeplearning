from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy
epsilon=numpy.finfo('float').eps
tf.set_random_seed(66)

# 1. 데이터
sess = tf.compat.v1.Session()
datasets = load_breast_cancer()
# x_data = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# y_data = pd.DataFrame(datasets.target, columns=['target'])
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1, 1)
print(x_data) # (569, 30)
print(y_data) # (569, 1)
print(x_data.dtype) # float64
print(y_data.dtype) # int32
# change type
# x_data = x_data.astype('float32')
# y_data = y_data.astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66, stratify=y_data)
sess.run(tf.compat.v1.global_variables_initializer())

x=tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y=tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w=tf.Variable(tf.compat.v1.random_normal([30,1]), name='weight')
b=tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

w=tf.compat.v1.random_normal([30,1])

# w=tf.Variable(tf.zeros([30,1]), name='weight')
# b=tf.Variable(tf.zeros([1]), name='bias')

# w=tf.Variable(tf.ones([30,1]), name='weight')
# b=tf.Variable(tf.ones([1]), name='bias')

# w dtype change to float64
# w = tf.cast(w, tf.float64)
# b = tf.cast(b, tf.float64)
# print(x_data[0:1])
# print(x_data[0:1].shape)


# print(tf.matmul(x_data[0:1], w))
# print(sess.run(tf.matmul(x_data[0:1], w)))


# 2. 모델구성


hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, w) + b)
# log1p : log(1+x) 계산
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 2001
for epochs in range(epoch) :
    cost_val, hy_val, _, w_val, b_val = sess.run([loss, hypothesis, train, w, b],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 100 == 0 :
        print(epochs, 'loss : ', cost_val, '\n')
        
# 4. 평가, 예측
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = np.where(y_pred > 0.5, 1, 0)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

sess.close()