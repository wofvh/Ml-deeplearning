import numpy as np
import tensorflow as tf
# print(tf.__version__)
# print(tf.executing_eagerly()) #True

# #즉시실행모드
# tf.compat.v1.disable_eager_execution() #disable 즉시실행모드를 끈다는 뜻 # 텐서2는 즉시실행모드 없음

# print(tf.executing_eagerly()) #False

# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)
# node3 = tf.add(node1, node2)


###################### 요기서부터 #################
sess = tf.compat.v1.Session()
a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)  # placeholder 여기서 정의해주는 코드

add_node = a + b

print(sess.run(add_node, feed_dict={a: 3, b: 4.5})) #7.5 # placeholder 사용하면 여기서 입력해줘야함
print(sess.run(add_node, feed_dict={a: [1, 3], b:[ 2,4]})) #[3,7]행렬로 연산됨 

add_and_triple = add_node * 3
print(add_and_triple)  #Tensor("mul:0", dtype=float32)

print(sess.run(add_and_triple, feed_dict={a:3 , b:4.5})) #22.5


# print(sess.run(add_node, feed_dict={a: 3, b: 4.5})) #7.5 # placeholder 사용하면 여기서 입력해줘야함
# print(sess.run(add_node, feed_dict={a: [1, 3], b:[ 2,4]})) #[3,7]행렬로 연산됨 
# import numpy as np
# import tensorflow as tf

class Mymodel():
    
    def data(self):
        self.a = tf.compat.v1.placeholder(tf.float32)
        self.b = tf.compat.v1.placeholder(tf.float32)
        self.triple = add_node * 3
        self.sess = tf.compat.v1.Session()
    
    def placeholder(self):
        self.add_node = self.a + self.b
        print(sess.run(add_node, feed_dict={a: 3, b: 4.5}))
        print(sess.run(add_node, feed_dict={a: [1, 3], b:[ 2,4]}))
        
        
         
        
amodel = Mymodel()
amodel.placeholder()