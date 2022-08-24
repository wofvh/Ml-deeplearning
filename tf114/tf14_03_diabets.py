#2,3,4,7,8,9,10,11,12
import tensorflow as tf
from sklearn import datasets
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
tf.set_random_seed(123)

datasets =  load_diabetes()

x_data = datasets.data
y_data = datasets.target

# sess = tf.compat.v1.InteractiveSession()
print(x_data.shape,y_data.shape)# x(442, 10)  (442,)


y_data = y_data.reshape(442,1)
print(y_data.shape) #(569, 1)

x_train , x_test , y_train , y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66,)

# print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.dtype, y_train.dtype) # (455, 30) (455, 1)

# y_train = np.array(y_train, dtype=np.int32)

print(x_train.shape, x_test.shape , y_train.shape, y_test.shape)
#(353, 10) (89, 10) (353, 1) (89, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
# w = tf.placeholder(tf.float32, shape=[30,1])
# b = tf.placeholder(tf.float32, shape=[1])
w = tf.compat.v1.Variable(tf.compat.v1.zeros([10,1]), name='weight')    # y = x * w  
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')   

hypothesis = tf.compat.v1.sigmoid (tf.matmul(x, w) + b) #예

# hypothesis = tf.sigmoid(hypothesis)
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for step in range(epoch):
    loss_val, hypothesis_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})  
    if step % 20== 0:
        print(epoch, 'loss : ', loss_val, '\n', hypothesis_val)
               
#4. 평가, 예측
y_predict = tf.cast(hypothesis > 0.5, dtype=tf.int32)  
# print(y_predict)   # Tensor("Cast:0", shape=(?, 1), dtype=float32)
# print(sess.run(hypothesis > 0.5, feed_dict={x:x_data, y:y_data}))

accuracy = tf.reduce_mean(tf.cast(tf.equal(y_data, y_predict), dtype=tf.float32))

predict, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print("===============================================================================")
print("예측값 : \n", hypothesis_val)
print("예측결과 : ", predict)
print("accuracy : ", acc)

sess.close()


# accuracy :  0.8681898