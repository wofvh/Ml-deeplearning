#행열연산 앞에서 행갯수와 곱해줄 열이 같아야함 if (4.5)*(5.3) = (4.3)shape

import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터

x_data = [[73, 51, 65],                       # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152],[185],[180],[205],[142]]      # (5, 1)                      

x = tf.compat.v1.placeholder(tf.float32, shape=[None,3]) #입력값 #placeholder를 이용해서 입력값을 받을 수 있다.
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1]) #입력값 #placeholder를 이용해서 입력값을 받을 수 있다.

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name = "weight") #초기값을 랜덤으로 생성해줌  
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name = "bias") #bias

hypothesis = tf.compat.v1.matmul(x, w) + b  #예측값  #hypothesis 가설  # y = x*w = b

# 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch =100

for epochs in range(epoch):
    _, loss_val, w_val = sess.run([train, loss, w], feed_dict={x:x_data, y:y_data})
    print(epoch, '\t', loss_val, '\t', w_val)
    
#4. 예측
predict = tf.matmul(x, w_val) + b   # predict = model.predict
y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error 
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, y_predict)
print('mae : ', mae)

# r2스코어 :  0.41260982355293097
# mae :  15.532891845703125