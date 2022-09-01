import tensorflow as tf 
tf.set_random_seed(123)

#1.데이터 
x_data = [[0,0],[0,1],[1,0],[1,1]]  # (4, 2)
y_data = [[0], [1], [1], [0]]       # (4, )


#2.모델구성
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1], name='weights'))
b = tf.Variable(tf.random_normal([1], name='bias'))

# 2.모델
h = tf.sigmoid(tf.matmul(x, w) + b)  

# 3-1.컴파일 
loss = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))               # binary cross entropy
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-6)
train = optimizer.minimize(loss)

# 3-2.훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train], feed_dict = {x:x_data, y:y_data})
    # if epochs %20 == 0:

print(epochs, 'loss : ', cost_val, hy_val)

# #4.평가, 예측
y_predict = sess.run(tf.cast(sess.run(h,feed_dict={x:x_data})>0.5, dtype=tf.float32))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
                                                                        #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_data,y_predict)
print('acc : ', acc)

sess.close()



