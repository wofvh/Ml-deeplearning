import tensorflow as tf
tf.compat.v1.set_random_seed(123)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] #6.2
y_data = [[0],[0],[0],[1],[1],[1]]             #6.1

x =tf.placeholder(tf.float32, shape = [None,2]) #6행 2열 이니깐 none.2
y = tf.placeholder(tf.float32, shape = [None,1]) #6행 1열 이니깐 none.1

w = tf.Variable(tf.random_normal([2,1]), name = 'weight') # x 랑 y 6행끼리 빼고 남은게 2행 1열 이니깐 2.1
b = tf.Variable(tf.random_normal([1]), name = 'bias') # bias 는 0 이니깐 1


# hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
hypothesis = tf.compat.v1.sigmoid (tf.matmul(x, w) + b) #예측값  #hypothesis 가설  # y = x*w = b
# model.add(Dense(1, activation='sigmoid'input_dim)) <<텐서 플로우 2 에서 이렇게 적용

# 컴파일 
# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) #binary cross entropy식 컴파일
#model.compile(loss='binary_crossentropy')<<텐서 플로우 2 에서 이렇게 적용


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #경사하강법 최적화 함수 #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다.
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch =100
for epochs in range(epoch):
    cost_val, hy_val, w_val = sess.run([loss, hypothesis, train],
                 feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epoch, 'loss : ', cost_val, '\n', hy_val)
    
#4. 예측
# predict = tf.matmul(x, w_val) + b   # predict = model.predict

y_predict = sess.run(tf.cast(hy_val>0.5, dtype=tf.float32)) #예측값이 0.5 이상이면 1 아니면 0

from sklearn.metrics import r2_score, mean_absolute_error , accuracy_score
acc = accuracy_score(y_data, y_predict)
print('r2스코어 : ', acc)

mae = mean_absolute_error(y_data, hy_val)
print('mae : ', mae)

sess.close()

# r2스코어 :  1.0
# mae :  0.2841140826543172