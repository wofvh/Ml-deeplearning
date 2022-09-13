# 앞뒤가 똑같은 autoencoder
# 딥하게 구성
# 2개의 모델을 구성하는데 하나는 기본적인 오토인코더
# 다른 하나는 딥하게 구성
# 2개 성능 비교

import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras import activations

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

#2. 모델
from keras.models import Model, Sequential
from keras.layers import Dense, Input

def autoencoder1(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder2(hidden1, hidden2, hidden3, hidden4):
    model = Sequential()
    model.add(Dense(units=hidden1, input_shape=(784,), activation='relu'))
    model.add(Dense(units=hidden2, activation='relu'))
    model.add(Dense(units=hidden3, activation='relu'))
    model.add(Dense(units=hidden4, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# model = autoencoder1(hidden_layer_size=32)
model = autoencoder2(64, 32, 32, 32)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, x_train, epochs=20)

#4. 평가, 예측
output = model.predict(x_test)

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))
    
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()
    