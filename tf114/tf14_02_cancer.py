#[실습] 
#2,3,4,7,8,9,10,11,12
import tensorflow as tf
from sklearn import datasets
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
tf.set_random_seed(123)

datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target

# sess = tf.compat.v1.InteractiveSession()
print(x_data.shape,y_data.shape)# x(569, 30) y(569,)

y_data = y_data.reshape(569,1)
print(y_data.shape) #(569, 1)

x_train , x_test , y_train , y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66, stratify=y_data)

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.dtype, y_train.dtype) # (455, 30) (455, 1)

# y_train = np.array(y_train, dtype=np.int32)

print(x_train.shape, x_test.shape , y_train.shape, y_test.shape)
#     (455, 30)     (114, 30)      (455, 1)       (114, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
# w = tf.placeholder(tf.float32, shape=[30,1])
# b = tf.placeholder(tf.float32, shape=[1])
w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,1]), name='weight')    # y = x * w  
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')   

hypothesis = tf.compat.v1.sigmoid (tf.matmul(x, w) + b) #예

# hypothesis = tf.sigmoid(hypothesis)
# hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs= 50        
for step in range(epochs):
     # sess.run(train)
      train_, loss_val, w_val, b_val = sess.run([train,loss,w,b],
                                       feed_dict={x_train:x_data, y_train:y_data}) #행렬을 받아서 실행하는 것이 아니라 피드데이터를 받아서 실행한다.
            # if step %2 == 0:
            #     print(step, loss_val, w_val, b_val)               

#4. 평가, 예측
y_predict = tf.cast(hypothesis >=0.5, dtype=tf.int32)  
y_predict = sess.run([hypothesis], feed_dict={x:x_test, y:y_test})

# acc = accuracy_score(y_train, np.round(y_predict))
# print("acc:", acc)
# mae = mean_squared_error(y_data, hy_val)
# print("mae", mae)

# sess.close()

# # 3-2. 훈련
# epochs = 5001
# for step in range(epochs):
#     _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
#     if step%20 == 0:
#         print(step, cost_val, hy_val)
        
# print('최종: ', cost_val, hy_val)

# # 4. 평가, 예측
# predict = tf.cast(hypothesis>=0.5, dtype=tf.float32)
# y_predict= sess.run([hypothesis], feed_dict={x:x_test, y:y_test})
# acc = accuracy_score(y_data, np.round(y_predict))
# print('acc: ', acc)

# mae = mean_squared_error(y_data, hy_val)
# print('mae: ', mae)

# sess.close()
