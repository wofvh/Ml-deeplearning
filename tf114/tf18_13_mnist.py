#[실습]
#DNN으로 MNIST 분류 shape None.784

from sys import float_repr_style
import tensorflow as tf
import keras
import numpy as np

tf.compat.v1.set_random_seed(123)

#1. 데이터
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)


from keras.utils import to_categorical

y_train  = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,784,1).astype('float32')/255.
x_test = x_test.reshape(10000,784,1).astype('float32')/255.

print(x_train.shape, y_train.shape) #(60000, 784, 1) (60000, 10)
print(x_test.shape, y_test.shape) #(10000, 784, 1) (10000, 10)


x = tf.compat.v1.placeholder(tf.float32, shape=[None,28,28,1]) #imput_shape
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10]) #output_shape

w1 = tf.compat.v1.get_variable('w1', shape=[2,1,64])#filter #1은 흑백, 3은 컬러 #64는 노드의 갯수
L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1], padding='SAME') #SAME은 패딩을 해준다. #VALID는 패딩을 안해준다. 
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L1_maxpool)



# # 3-1.컴파일 
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))     # categorical_crossentropy
# optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
# train = optimizer.minimize(loss)

# # 3-2.훈련 
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# epoch = 100
# for epochs in range(epoch):
#     cost_val, hy_val, _  = sess.run([loss, h, train], feed_dict = {x:x_train, y:y_train})
#     if epochs %20 == 0:

#         print(epochs, 'loss : ', cost_val, 'hy_val : ', hy_val)


# # #4.평가, 예측
# y_predict = sess.run(h,feed_dict={x:x_test})
# y_predict = sess.run(tf.argmax(y_predict,axis=1))           #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
# y_test = sess.run(tf.argmax(y_test,axis=1))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
#                                                                        #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

# from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

# acc = accuracy_score(y_test,y_predict)
# print('acc : ', acc)

# sess.close()
