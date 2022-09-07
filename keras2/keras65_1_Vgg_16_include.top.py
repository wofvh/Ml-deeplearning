import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications import VGG16

model = VGG16() # vgg16은 이미지 분류 모델 
model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) # include_top=False : 이미지 분류 모델에서는 1000개의 노드가 있지만 우리는 10개의 노드만 필요하므로 False로 설정

model.summary()
print(len(model.weights))               #26
print(len(model.trainable_weights))     #26 #trainable_weights는 훈련가능한 가중치 #trainable_weights
