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
# encoded = Dense(64, activation='relu')(input_img) # 노드를 줄이는 이유는 불필요한 요소 제거후 다시 늘림
# encoded = Dense(1064, activation='relu')(input_img) # 노드를 늘릴경우는 더 많은 정보를 압축시킴
# encoded = Dense(16, activation='relu')(input_img) # relu 는 0 부터 무한데까지의 값을 가짐
encoded = Dense(1064, activation='relu')(input_img) #
# decoded = Dense(784, activation='relu')(encoded) #sigmoid를 쓰는이유는 0~1사이의 값으로 나오게 하기 위해서 제구성시 스케일링을 했기 때문에
# decoded = Dense(784, activation='linear')(encoded)  
decoded = Dense(784, activation='sigmoid')(encoded)  

autoencoder = Model(input_img, decoded) # autoencoder란 인풋과 아웃풋이 같은 모델

# autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc']) 
autoencoder.compile(optimizer='adam', loss='mse',metrics=['acc']) 

autoencoder.fit(x_train, x_train, epochs=20, batch_size=35, # x로 x를 훈련시킴!
                validation_split=0.2) #중지도학습 이기 때문에 레이블이 필요없음

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()










