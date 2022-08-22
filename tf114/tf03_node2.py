# import tensorflow as tf 

# node1 = tf.constant(3.0)
# node2 = tf.constant(4.0)
# # node3 = node1 + node2
# node3 = tf.add(node1, node2)

# # print(node3)

# # Tensor("add:0", shape=(), dtype=float32) < 알림으로 나옴 sess 없을때 sess.run 사용해야함
# # sess = tf.Session()

# sess = tf.compat.v1.Session()
# print(sess.run(node3))           #7.0

######################################


import tensorflow as tf 

node1 = tf.constant(10.0)
node2 = tf.constant(2.0)
#덧셈 node3
#뺼셈 node4
#곱샘 node5
#나누셈 node6

#node3 = node1 + node2
# node3 = tf.add(node1, node2)
# node4 = tf.subtract(node1, node2) #뺼셈 node4
node5 = tf.multiply(node1, node2) #곱샘 node5
# node6 = tf.divide(node1, node2)   #나누셈 node6


sess = tf.compat.v1.Session()
print(sess.run(node5))   


