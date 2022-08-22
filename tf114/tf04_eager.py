import tensorflow as tf 
print(tf.__version__)
print(tf.executing_eagerly()) #False

tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly()) #False
