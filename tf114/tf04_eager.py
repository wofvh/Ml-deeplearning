import tensorflow as tf 
print(tf.__version__)
print(tf.executing_eagerly()) #False

#즉시실행모드!!
tf.compat.v1.disable_eager_execution() #disable 즉시실행모드를 끈다는 뜻 # 텐서2는 즉시실행모드 없음 

# print(tf.executing_eagerly()) #False

hello = (tf.constant("Hello, world!")) #헬로우라는 그래프를 정의 #constant 변하지 않는 상수 

sess = tf.compat.v1.Session()
print(sess.run(hello)) #b'Hello, world!'바이너리 형태 출력
