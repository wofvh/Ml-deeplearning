from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Conv2D , Flatten,MaxPool2D

#1.데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,)
    
print(x_train.shape, y_train.shape)    #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      #(10000, 32, 32, 3) (10000, 1)
    
augument_size = 20000
randidx = np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()  #(20000, 32, 32, 3)
y_augumented = y_train[randidx].copy()  #(20000, 1)


print(x_augumented.shape) # (4000, 32, 32, 3)
print(y_augumented.shape) # (4000, 1)


# x_train = x_train.reshape(40000,32,32,1)
x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2], 3)

x_augumented = train_datagen.flow(x_augumented,y_augumented,
                                  batch_size = augument_size,
                                  shuffle=False).next()[0]

x_train =np.concatenate((x_train, x_augumented))
y_train =np.concatenate((y_train, y_augumented))


print(x_train.shape) # (70000, 32, 32, 3)
print(y_train.shape) #(70000, 1)


x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

xy_df2 = train_datagen.flow(x_train,y_train,
                                  batch_size=augument_size,shuffle=False)
x_df = np.concatenate((x_train,x_augumented))
y_df = np.concatenate((y_train,y_augumented))
# print(x_df.shape) #(64000, 28, 28, 1)

xy_df3 = test_datagen.flow(x_df,y_df,
                       batch_size=augument_size,shuffle=False)



x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

xy_df2 = train_datagen.flow(x_train,y_train,
                                  batch_size=augument_size,shuffle=False)
x_df = np.concatenate((x_train,x_augumented))
y_df = np.concatenate((y_train,y_augumented))
# print(x_df.shape) #(64000, 28, 28, 1)

xy_df3 = test_datagen.flow(x_df,y_df,
                       batch_size=augument_size,shuffle=False)



# print(xy_df3[0].shape) #(4000, 28, 28, 1)
# print(xy_df3[0][1].shape) #(4000,)

# 소괄호() 1개와  소괄호(()) 2개의 차이를 공부해라! 2개인 이유는 안에 옵션을 더 넣을 수 있기 때문이다.아무 것도 안하면 디폴트로 들어감
print(x_train.shape,y_train.shape) #(100000, 28, 28, 1) (100000,)


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from keras.datasets import mnist,cifar10
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np


#1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,)
    

augument_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()


print(x_augumented.shape)  #(400, 28, 28)
print(y_augumented.shape) #(400,) 50000, 32, 32, 3)
print(x_train.shape)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2], 3)



xy_df2 = train_datagen.flow(x_train,y_train,
                                  batch_size=augument_size,shuffle=False)
x_df = np.concatenate((x_train,x_augumented))
y_df = np.concatenate((y_train,y_augumented))
# print(x_df.shape) #(64000, 28, 28, 1)

xy_df3 = test_datagen.flow(x_df,y_df,
                       batch_size=augument_size,shuffle=False)
from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test =train_test_split(xy_df3[0][0],xy_df3[0][1],train_size=0.75,shuffle=False)
print(x_test.shape)
print(y_test.shape)

'''
