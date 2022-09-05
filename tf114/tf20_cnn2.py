import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

#1. 데이터
x_train = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]])

print(x_train.shape)  #(1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3, 3, 1]) #input_shape #none 행무시

w = tf.compat.v1.constant([[[[1.]],[[1.]]],
                           [[[1.]],[[1.]]]]) #filter #커널사이즈, 컬러, 필터

print(w) #Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)  #(2, 2, 1, 1) <커널사이즈 / 컬러 필터>


# L1 = tf.nn.conv2d(x,w,strides=[1,1,1,1], padding='SAME') #SAME은 패딩을 해준다. #VALID는 패딩을 안해준다
L1 = tf.nn.conv2d(x,w,strides=[1,2,2,1], padding='VALID') #SAME은 패딩을 해준다. #VALID는 패딩을 안해준다. #srpides는 커널사이즈와 같다. #stride는 커널을 몇칸씩 움직일지 결정한다.
print(L1)               #Tensor("Conv2D:0", shape=(?, 2, 2, 1), dtype=float32)


sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={x:x_train})
print('============================결과=============')
print(output) #[[[[12.][16.]] [[24.][28.]]]] 
print('============================결과=============')
print(output.shape) #(1, 2, 2, 1)  pading 후 (1, 3, 3, 1)