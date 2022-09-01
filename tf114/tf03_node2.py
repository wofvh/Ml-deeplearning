import tensorflow as tf 

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

# print(node3)

# Tensor("add:0", shape=(), dtype=float32) < 알림으로 나옴 sess 없을때 sess.run 사용해야함
# sess = tf.Session()

sess = tf.compat.v1.Session()
print(sess.run(node3))           #7.0

# #####################################



# node1 = tf.constant(10.0)
# node2 = tf.constant(2.0)
# # #덧셈 node3
# #뺼셈 node4
# #곱샘 node5
# #나누셈 node6

# #node3 = node1 + node2
# node3 = tf.add(node1, node2)
# node4 = tf.subtract(node1, node2) #뺼셈 node4
# node5 = tf.multiply(node1, node2) #곱샘 node5
# node6 = tf.divide(node1, node2)   #나누셈 node6


# sess = tf.compat.v1.Session()
# print(sess.run(node3))   



import tensorflow as tf 

class Add():
    def __init__(self):
        self.node1 = tf.constant(10.0)
        self.node2 = tf.constant(10.0)
        self.sess = tf.compat.v1.Session()
        
    def add(self):
        self.node3 = tf.add(self.node1, self.node2)
        print(self.sess.run(self.node3))   
        
    def subtract(self):
        self.node3 =  tf.subtract(self.node1, self.node2)
        print(self.sess.run(self.node3)) 
        
    def multiply(self):
        self.node3 =  tf.multiply(self.node1, self.node2)
        print(self.sess.run(self.node3)) 
        
    def divide(self):
        self.node3 =  tf.divide(self.node1, self.node2)
        print(self.sess.run(self.node3)) 
        
    

        
        
a = Add()
a.add()
a.subtract()
a.multiply()
a.divide()

# class 객체선공!
# 20.0
# 0.0 
# 100.0
# 1.0