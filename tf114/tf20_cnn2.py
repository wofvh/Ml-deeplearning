import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

#1. 데이터
x_train = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]])

print(x_train.shape)  #(1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3, 3, 1]) #imput_shape #none 행무시

w = tf.compat.v1.constant([[[[1.]],[[1.]]],
                           [[[1.]],[[1.]]]]) #filter #커널사이즈, 컬러, 필터