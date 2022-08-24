from numpy import dtype
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]    # (6,2)
y_data = [[0],[0],[0],[1],[1],[1]]                     # (6,1)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.compat.v1.Variable(tf.random.normal([2,1]), name = 'weight1')   # (2,1)
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델구성
# hypothesis = x * w + b
# hypothesis = tf.matmul(x, w) + b
# hypothesis = tf.sigmoid(hypothesis)
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
# model.add(Dense(1, activation='sigmoid'))

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   
# binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(2001):
    loss_val, hypothesis_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})  
    if epoch % 200 == 0:
        print(epoch, 'loss : ', loss_val, '\n', hypothesis_val)
               
#4. 평가, 예측
y_predict = tf.cast(hypothesis > 0.5, dtype=tf.int32)  # tf.cast('조건') 함수 : True이면 1, False이면 0을 출력한다. 텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다. 
# print(y_predict)   # Tensor("Cast:0", shape=(?, 1), dtype=float32)
# print(sess.run(hypothesis > 0.5, feed_dict={x:x_data, y:y_data}))


accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print("===============================================================================")
print("예측값 : \n", hypothesis_val)
print("예측결과 : ", pred)
print("accuracy : ", acc)

sess.close()

# ===============================================================================
# 예측값 :
#  [[0.04392562]
#  [0.17360151]
#  [0.36277783]
#  [0.7560311 ]
#  [0.92249364]
#  [0.9746191 ]]
# 예측결과 :  [[0.]
#  [0.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]]
# accuracy :  1.0


# predict = tf.matmul(x, w_val) + b   # predict = model.predict

# y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})
# print("예측 : " , y_predict)

# sess.close()

# from sklearn.metrics import r2_score, mean_absolute_error
# r2 = r2_score(y_data, y_predict)
# print('r2스코어 : ', r2)

# mae = mean_absolute_error(y_data, y_predict)
# print('mae : ', mae)



    