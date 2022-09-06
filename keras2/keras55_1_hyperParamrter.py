import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Input
from tensorflow.python.keras.models import sequential
import tensorflow as tf
print(tf.__version__)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

