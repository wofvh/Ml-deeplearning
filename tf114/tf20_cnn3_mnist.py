import tensorflow as tf
import keras
import numpy as np

tf.compat.v1.set_random_seed(123)

#1. 데이터
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train  = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None,28,28,1]) #imput_shape
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10]) #output_shape

#3. 컴파일, 훈련
w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,128])#filter #1은 흑백, 3은 컬러 #64는 노드의 갯수
                #커널싸이즈, 컬러, 필터
L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1], padding='SAME') #SAME은 패딩을 해준다. #VALID는 패딩을 안해준다. 
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# model.add(Conv24D(64,kernel_size = (2,2), input_shape=(28,28,1))) #커널사이즈, 컬러, 필터

print(w1) #<tf.Variable 'w1:0' shape=(2, 2, 1, 128) dtype=float32_ref> #커널사이즈, 컬러, 필터
print(L1) #Tensor("Relu:0", shape=(?, 28, 28, 128), dtype=float32)    #패딩을 안해준다.
print(L1_maxpool)

w2 = tf.compat.v1.get_variable('w2', shape=[3,3,128,64])#filter #1은 흑백, 3은 컬러 #64는 노드의 갯수
L2 = tf.nn.conv2d(L1_maxpool,w2,strides=[1,1,1,1], padding='VALID') #SAME은 패딩을 해준다. #VALID는 패딩을 안해준다. 
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L2)  #Tensor("MaxPool:0", shape=(?, 14, 14, 128,dtype=float32)
print(L2_maxpool) #Tensor("Selu:0", shape=(?, 12, 12, 64), dtype=float32)


w3 = tf.compat.v1.get_variable('w3', shape=[3,3,64,32])#filter #1은 흑백, 3은 컬러 #64는 노드의 갯수
L3 = tf.nn.conv2d(L2_maxpool,w3,strides=[1,1,1,1], padding='VALID') #SAME은 패딩을 해준다. #VALID는 패딩을 안해준다. 
L3 = tf.nn.elu(L3)
# L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L3) #Tensor("Elu:0", shape=(?, 4, 4, 32), dtype=float32)

#flatten
L_flat = tf.reshape(L3, [-1, 4*4*32])
print('플래튼:',L_flat)


w4 = tf.get_variable('w4', shape=[4**32, 100], initializer=tf.contrib.
                     layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([100]), name='b5')
L4 = tf.nn.selu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=0.7)  #rate=0.3 #0.7은 30%를 끈다. #0.3은 70%를 끈다.


w5 = tf.get_variable('w5', shape=[100, 10], initializer=tf.contrib.
                     layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name='b5')
L5 = tf.matmul(L4, w5) + b5
hyporthesis = tf.nn.softmax(L5)

print(hyporthesis) #Tensor("Softmax:0", shape=(?, 10), dtype=float32)

#3. 컴파일, 훈련
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hyporthesis), axis=1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L5, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 3-2.훈련 

training_epochs = 30
batch_size = 100
total_batch = int(len(x_train)/batch_size)  #60000/100 = 600 #1에폭당 600번 돈다.#int는 소수점을 버린다.
print(total_batch)

for epoch in range(training_epochs):
    
    for i in range(total_batch):  #600번 돈다.
        start = i* batch_size   #0
        end = start + batch_size #100
        batch_x, batch_y = x_train[start:end], y_train[start:end] #x_train[0:100], y_train[0:100] #0~99번째 까지
        
        feed_dict = {x:batch_x, y:batch_y} #x_train, y_train
        batch_loss, _, = sess.run([loss, optimizer], feed_dict=feed_dict)   #loss, optimizer를 실행시키겠다. #_는 필요없는 값을 의미한다.
        
        avg_loss = batch_loss/total_batch #평균 loss = batch_loss/600 #1에폭당 loss #600번 돌면서 loss를 구한다.
        
        
    print("에포:,")
# for epochs in range(epoch):
#     cost_val, hy_val, _  = sess.run([loss, hyporthesis, train], feed_dict = {x:x_train, y:y_train})
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
