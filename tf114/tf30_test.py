from sklearn import datasets
from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)    #(442, 10) (442,)

y_data = y_data.reshape(442,1)
print(y_data.shape)   # (442, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)   # (354, 13) (354, 1)
print(x_test.shape, y_test.shape)     # (152, 13) (152, 1)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.compat.v1.Variable(tf.random.normal([10,1]), name='weight')    # y = x * w  
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')   

#2. 모델구성
# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000995)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    _, loss_val, w_val= sess.run([train, loss, w], feed_dict={x:x_train, y:y_train})
    print(epochs, '\t', loss_val, '\t', w_val)
           
#4. 예측
predict = tf.matmul(x, w_val) + b   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x:x_test, y:y_test})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)

# r2스코어 :  0.02105369681810465
# mae :  68.48492225904143