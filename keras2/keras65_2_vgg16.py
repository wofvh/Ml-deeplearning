import numpy as np
from keras.models import Sequential
from keras.layers import Dense ,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.applications import VGG16 ,


# model = VGG16() # vgg16은 이미지 분류 모델 #include_top = False : 이미지 분류 모델에서는 마지막 레이어를 제외하고 사용 # if include_top = True , input_shape = (224, 224, 3) 
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) # include_top=False : 이미지 분류 모델에서는 1000개의 노드가 있지만 우리는 10개의 노드만 필요하므로 False로 설정

# vgg16.summary()
# vgg16.trainable=False # 가중치를 동결한다. # vgg16.trainable=True # vgg16을 훈련시킨다. 
# vgg16.summary()

model = Sequential()
model.add(vgg16) 
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

model.trainable = False
                                #trainable:True /vgg16.trainable=False /# model False
model.summary()
print(len(model.weights))           #30 / 30 /30
print(len(model.trainable_weights)) #30 / 4  / 0   # model False          