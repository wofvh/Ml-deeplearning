# y = wx = b
from cProfile import run
import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터 
x = tf.placeholder(tf.float32, shape=[None]) #입력값 #placeholder를 이용해서 입력값을 받을 수 있다.
y = tf.placeholder(tf.float32, shape=[None]) #입력값 #placeholder를 이용해서 입력값을 받을 수 있다.
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #초기값을 랜덤으로 생성해줌
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #초기값을 랜덤으로 생성해줌
#2. 모델 구성
hypothesis = x * w + b  #예측값  #hypothesis 가설  # y = x*w = b
#3-1. 컴파일
loss  = tf.reduce_mean(tf.square(hypothesis - y))#square 제곱 #손실함수 #오차의 제곱의 평균을 손실함수로 정의한다 .#mse = mean squared error
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #경사하강법 최적화 함수 #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다.
train = optimizer.minimize(loss) #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다. #loss를 최소화하는 최적화 함수
#3-2. 훈련
with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer()) #변수를 초기화해줌
     
        epochs = 2001   #에포(훈련) 수    
        for step in range(epochs):
            # sess.run(train)
            train_,loss_val, w_val, b_val = sess.run([train,loss,w,b],
                                       feed_dict={x:[1,2,3,4,5], y:[1,2,3,4,5]}) #행렬을 받아서 실행하는 것이 아니라 피드데이터를 받아서 실행한다.
            if step %20 == 0:
                print(step, loss_val, w_val, b_val)

# sess.close()

