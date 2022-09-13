    #autoencoder gen이나오기 전까지 킹왕짱 하던놈
from pyexpat import model
import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

print(_.shape)#(10000,)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255


from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size,input_shape=(784,), 
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# model = autoencoder(hidden_layer_size=154) #PCA의 95%성능 

model = autoencoder(hidden_layer_size=331) #PCA의 95%성능 

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train, x_train, epochs=5) # x로 x를 훈련시킴!)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig,((ax1 ,ax2 ,ax3, ax4, ax5),(ax6, ax7 ,ax8,ax9,ax10)) = \
        plt.subplots(2,5, figsize=(20,7))

#이미지 다섯개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]),5)

#원본(입력) 이미지를 맨위에 그린다
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28))
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])    
    
#오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28))
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()