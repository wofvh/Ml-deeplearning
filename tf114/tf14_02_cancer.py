#[실습] 
#2,3,4,7,8,9,10,11,12
import tensorflow as tf
from sklearn import datasets
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error
tf.set_random_seed(123)

datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target

# sess = tf.compat.v1.InteractiveSession()
print(x_data.shape,y_data.shape)# x(569, 30) y(569,)
y_data = y_data.reshape(-1,1)
print(y_data.shape) #(569, 1)

x_train , x_test , y_train , y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66, stratify=y_data)

# print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(x_train.dtype, y_train.dtype) # (455, 30) (455, 1)

# # y_train = np.array(y_train, dtype=np.int32)

print(x_train.shape, x_test.shape , y_train.shape, y_test.shape)
#     (455, 30)     (114, 30)      (455, 1)       (114, 1)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None,1])
# w = tf.placeholder(tf.float32, shape=[30,1])
# b = tf.placeholder(tf.float32, shape=[1])
w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,1]), name='weight')    # y = x * w  
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')   

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b) #예

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y_train*tf.log(hypothesis)+(1-y_train)*tf.log(1-hypothesis))  # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs= 2001   
for step in range(epochs):
     # sess.run(train)
    loss_val, hy_val, _, w_val, b_val, = sess.run([loss, hypothesis, train, w, b,],
                                        feed_dict={x:x_train,y:y_train}) #행렬을 받아서 실행하는 것이 아니라 피드데이터를 받아서 실행한다.
    if step %20 ==0:
        print(epochs, '\t', 'loss:',loss_val, '\t', hy_val)       

# 4. 평가, 예측
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = np.where(y_pred > 0.5, 1, 0)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

sess.close()


# acc :  0.9385964912280702