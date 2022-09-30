import tensorflow as tf 
tf.set_random_seed(123)

#1.데이터 
x_data = [[0,0],[0,1],[1,0],[1,1]]  # (4, 2)
y_data = [[0], [1], [1], [0]]       # (4, )

#2.모델구성
## input layer ##
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

## hidden layer ##
w1 = tf.Variable(tf.random_normal([2, 20], name='weights'))
b1 = tf.Variable(tf.random_normal([20], name='bias'))

h1 = tf.sigmoid(tf.matmul(x, w1) + b1)    # 여기서 나온 h와 아웃풋 레이어와 연산됨.

## output layer ##
w2 = tf.Variable(tf.random_normal([20, 1], name='weights'))
b2 = tf.Variable(tf.random_normal([1], name='bias'))

h = tf.sigmoid(tf.matmul(h1, w2) + b2)  

# 3-1.컴파일 
loss = -tf.reduce_mean(y*tf.log(h)+(1-y)*tf.log(1-h))               # binary cross entropy
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# 3-2.훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 50000
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




