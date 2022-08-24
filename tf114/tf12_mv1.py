import tensorflow as tf
tf.compat.v1.set_random_seed(1234)

#1. 데이터
         # 첫번 두번 세번 네번 다섯번
x1_data = [73., 93., 89., 96., 73.]       # 국어                  .을 찍은 이유는 float형태로 나타내주기위해서
x2_data = [80., 88., 91., 98., 66.]       # 영어
x3_data = [75., 93., 90., 100., 70.]      # 수학
y_data = [152., 185., 180., 196., 142.]   # 환산점수

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight1')
w2 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight2')
w3 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight3')
b = tf.compat.v1.Variable(tf.random.normal([1]), name="bias1")

#2. 모델구성
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) #경사하강법 최적화 함수 #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다.
train = optimizer.minimize(loss) #최적화 함수를 이용해서 손실함수의 미분값을 최소화한다. #loss를 최소화하는 최적화 함수


#훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2000
for epochs in range(epoch):
    cost_val,hy_val,_ = sess.run([loss,hypothesis,train ],
         feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if epochs % 20 == 0:
       print(epochs, "loss:", cost_val, "\n", hy_val)    
sess.close()
       
#4. 예측

# y_predict = sess.run(predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y_data:y_data})
# print("예측 : " , y_predict)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, hy_val)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, hy_val)
print('mae : ', mae)


# r2스코어 :  0.999246000805042
# mae :  0.4440338134765625

