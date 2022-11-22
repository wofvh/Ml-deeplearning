import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications import VGG16

# model = VGG16() # vgg16은 이미지 분류 모델 #include_top = False : 이미지 분류 모델에서는 마지막 레이어를 제외하고 사용 # if include_top = True , input_shape = (224, 224, 3) 
model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) # include_top=False : 이미지 분류 모델에서는 1000개의 노드가 있지만 우리는 10개의 노드만 필요하므로 False로 설정

model.summary()
print(len(model.weights))               #26
print(len(model.trainable_weights))     #26 #trainable_weights는 훈련가능한 가중치 #trainable_weights

#############################include_top = True ###########################
#1. FC .layer 원래꺼 그대로사용 
#2. input_shape = (224,224,3)고정값. 으로 변경할수 없다
# =================================================================
#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0

#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792

#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
#############################include_top = False ###########################
# input_1 (InputLayer)        [(None, 32, 32, 3)]       0

#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792

#  block1_conv2 (Conv2D)       (None, 32, 32, 64)        36928

#'''''''''''''''''''''''''''''''''''' 플래튼 하단이 없어짐 ''''''''''''''''''''''''

#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0