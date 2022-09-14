#[실습]keras47_4 male , fmale noise를 넣어서 
#predict 첫번째 : 기미 주근깨 여드름 제거 
#predict 두번째: 본인 사진너서 원본수정 !
#noise 넣어서 수정 

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D,MaxPooling2D,UpSampling2D,Flatten
from bitarray import test
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False


x_train = np.load('d:/study_data/_save/_npy/men_women_3_train_x.npy')
x_test = np.load('d:/study_data/_save/_npy/men_women_3_test_x.npy')
y_train = np.load('d:/study_data/_save/_npy/men_women_3_train_y.npy')
y_test = np.load('d:/study_data/_save/_npy/men_women_3_test_y.npy')
a_test = np.load('d:/study_data/_save/_npy/keras_test_a.npy')


x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
a_test_noised = a_test + np.random.normal(0, 0.1, size=x_test.shape)


x_train_noised = np.clip(x_train_noised , a_min=0, a_max=1) #0이하는 1로 바뀜 
x_test_noised = np.clip(x_test_noised , a_min=0, a_max=1) #0이하는 1로 바뀜
a_test_noised = np.clip(a_test_noised , a_min=0, a_max=1) #0이하는 1로 바뀜  

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same',strides=2, input_shape=(150,150,3)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same' ))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same',strides=2, input_shape=(150,150,3)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    # model.compile(optimizer='rmsprop', loss='mse')
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = autoencoder(hidden_layer_size=320)
# pca를 통해 0.95 이상인 n_component  몇개?
# 0.95  # 154
# 0.99  # 331
# 0.999 # 486
# 1.0   # 713
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train_noised, x_train, epochs=30, batch_size=256,
                validation_split=0.2)
output = model.predict(x_test)
output2 = model.predict(a_test)

from matplotlib import pyplot as plt
import random 

fig, ((ax1, ax2, ax3, ax4, ax5,image01), (ax6, ax7, ax8, ax9, ax10,image02),
      (ax11, ax12, ax13, ax14, ax15,image03))  = \
    plt.subplots(3, 6, figsize=(20, 7))
    
# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5,]):
    ax.imshow(x_test[random_images[i]].reshape(150,150,3), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
image01.imshow(a_test[0])  
             
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150,150,3), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])             
image02.imshow(a_test_noised[0])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(150,150,3), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
image03.imshow(output2[0])
        
plt.tight_layout()
plt.show()  