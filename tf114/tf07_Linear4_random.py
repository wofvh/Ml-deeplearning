# y = wx = b
from cProfile import run
import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터 
x = [1, 2, 3, 4 ,5] 
y = [1, 2, 3, 4 ,5] 

# w = tf.Variable(1111, dtype=tf.float32)
# b = tf.Variable(11, dtype=tf.float32) #b = <bias
W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #초기값을 랜덤으로 생성해줌
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #초기값을 랜덤으로 생성해줌

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) #변수를 초기화해줌
print(sess.run(W))  #[-1.5080816]
'''
#2. 모델 구성
hypothesis = x * w + b  #예측값  #hypothesis 가설  # y = x*w = b


#3-1. 컴파일
loss  = tf.reduce_mean(tf.square(hypothesis - y))#square 제곱 #손실함수 #오차의 제곱의 평균을 손실함수로 정의한다 .#mse = mean squared error

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #경사하강법 최적화 함수 #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다.
train = optimizer.minimize(loss) #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다. #loss를 최소화하는 최적화 함수
#model.compile(loss='mse', optimizer='agd')


#3-2. 훈련
with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer()) #변수를 초기화해줌
     
        epochs = 2001   #에포(훈련) 수    
        for step in range(epochs):
            sess.run(train)
            if step %4 == 0:
                print(step,sess.run(loss) ,sess.run(w),sess.run(b))

# sess.close()
'''