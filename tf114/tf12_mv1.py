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


'''
#########################################################################################################

#2. 모델 구성
#3-1. 컴파일
#3-2. 훈련

        sess.run(tf.compat.v1.global_variables_initializer()) #변수를 초기화해줌
     
        epochs = 2001   #에포(훈련) 수    
        for step in range(epochs):
            # sess.run(train)
            train_, loss_val, w_val, b_val = sess.run([train,loss,w,b],
                                       feed_dict={x_trian:[1,2,3,4,5], y_train:[1,2,3,4,5]}) #행렬을 받아서 실행하는 것이 아니라 피드데이터를 받아서 실행한다.
            if step %2 == 0:
                print(step, loss_val, w_val, b_val)

#############################################################################################
x_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32,shape = [None])
y_perdict = x_test * w_val + b_val          #y_perdict  = model.predict(x_test)
sess = tf.compat.v1.Session()#<<sess.run해주기전 필수로 불러와줘야함 !
print("[6,7,8]의 예측값 : ", sess.run(y_perdict, feed_dict={x_test:x_data}))#y_perdict  = model.predict(x_test) + 예측할값
sess.close()




###########################################################################################

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(10001):
    _, loss_val, w_val1, w_val2, w_val3 = sess.run([train, loss, w1, w2, w3], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    print(epochs, '\t', loss_val, '\t', w_val1, '\t', w_val2, '\t', w_val3)
    
   
#4. 예측
predict =  x1*w_val1 + x2*w_val2 + x3*w_val3 + b   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, y_predict)
print('mae : ', mae)
'''