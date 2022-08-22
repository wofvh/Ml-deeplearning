import tensorflow as tf 
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32) #변수를 정의해줌 
y = tf.Variable([3], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer() #변수를 초기화해줌
sess.run(init)  #변수를 초기화해줌 초기값을 넣어줄수있는 상태 #초기화 시키고 실행시켜야함 

print(sess.run(x+y)) #5.0