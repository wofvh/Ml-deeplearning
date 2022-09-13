    #autoencoder gen이나오기 전까지 킹왕짱 하던놈

import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

print(_.shape)#(10000,)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255


from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) # 노드를 줄이는 이유는 불필요한 요소 제거후 다시 늘림

decoded = Dense(784, activation='sigmoid')(encoded) #sigmoid를 쓰는이유는 0~1사이의 값으로 나오게 하기 위해서 제구성시 스케일링을 했기 때문에

autoencoder = Model(input_img, decoded) # autoencoder란 인풋과 아웃풋이 같은 모델

# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

autoencoder.fit(x_train, x_train, epochs=50, batch_size=35, # x로 x를 훈련시킴!
                validation_split=0.2) #중지도학습 이기 때문에 레이블이 필요없음














