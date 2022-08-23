import tensorflow as tf 
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight') #초기값을 랜덤으로 생성해줌

#1. 초기화 첫번째 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa:', aaa)  #aaa: [-1.5080816]
sess.close()

#2. 초기화 두번째 방법
sess =  tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess) # << 변수 초기화하는 2번째 방법
print("bbb:", bbb) 
sess.close()

#.3 초기화 세번째 방법
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval() # << 변수 초기화하는 3번째 방법  InteractiveSessio는 eval 하고 변수를 안 넣어도됨 
print("ccc:", ccc)
sess.close()


eval()