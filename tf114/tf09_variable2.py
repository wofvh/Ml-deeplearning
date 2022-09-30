#실습
#lr 수정해서 epoch를 100이하로 바꿔서 실행해보기
#step = 100 이하, w = 1.99, b = 0.99
import tensorflow as tf
tf.set_random_seed(123)
#1. 데이터 
x_trian_data = [1, 2 , 3] 
y_trian_data = [1, 5 , 7]
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #초기값을 랜덤으로 생성해줌
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #초기값을 랜덤으로 생성해줌

x_train = tf.placeholder(tf.float32, shape=[None]) #입력값 #placeholder를 이용해서 입력값을 받을 수 있다.
y_train = tf.placeholder(tf.float32, shape=[None])  
x_test = tf.placeholder(tf.float32, shape=[None])  


#2. 모델 구성
hypothesis = x_train * w + b  #예측값  #hypothesis 가설  # y = x*w = b

#3-1. 컴파일
loss  = tf.reduce_mean(tf.square(hypothesis - y_train))#square 제곱 #손실함수 #오차의 제곱의 평균을 손실함수로 정의한다 .#mse = mean squared error
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.10) #경사하강법 최적화 함수 #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다.
train = optimizer.minimize(loss)  # optimizer='sgd'

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) #변수를 초기화해줌
     
epochs= 50        
for step in range(epochs):
     # sess.run(train)
      train_, loss_val, w_val, b_val = sess.run([train,loss,w,b],
                                       feed_dict={x_train:x_trian_data, y_train:y_trian_data}) #행렬을 받아서 실행하는 것이 아니라 피드데이터를 받아서 실행한다.
            # if step %2 == 0:
            #     print(step, loss_val, w_val, b_val)

x_test_data = [6,7,8]
y_predict = x_test * w_val + b_val

print("[6,7,8]예측 : ", sess.run(y_predict, feed_dict={x_test:x_test_data}))#y_perdict  = model.predict(x_test) + 예측할값
sess.close()

# [6,7,8]의 예측값 :  [13.874613 16.109297 18.34398 ]

############################################### 2. Session() // eval #####################################################

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1000
for step in range(epochs):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                        feed_dict={x_train:x_trian_data, y_train:y_trian_data})
    # if step % 20 == 0:
    #     # print(step, sess.run(loss), sess.run(w), sess.run(b))
    #     print(step, loss_val, w_val, b_val)
    
#4. 예측
x_test_data = [6,7,8]
predict = x_test * w_val + b_val      # predict = model.predict

print("[6,7,8] 예측 : " , predict.eval(session=sess, feed_dict={x_test:x_test_data}))

sess.close()
############################################### 3. InteractiveSession() // eval #####################################################

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1000
for step in range(epochs):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                        feed_dict={x_train:x_trian_data, y_train:y_trian_data})
    # if step % 20 == 0:
    #     # print(step, sess.run(loss), sess.run(w), sess.run(b))
    #     print(step, loss_val, w_val, b_val)
    
#4. 예측
x_test_data = [6,7,8]
predict = x_test * w_val + b_val      # predict = model.predict

print("[6,7,8] 예측 : " , predict.eval(feed_dict={x_test:x_test_data}))

sess.close()
