import tensorflow as tf 
print(tf.__version__)

# print("Hello, world!")
#텐서 플로우에는 함수 변수 constan라는 놈이있음

hello = (tf.constant("Hello, world!"))
# print(hello)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))  #b'Hello, world!'바이너리 형태 출력

#텐서플로우는 출력을할때 반드시 sess.run을 실행해야함 <텐서플로 2 부터sess.run 필요없음 

