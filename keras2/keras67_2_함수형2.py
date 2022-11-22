from keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras.layers import Dense, Input ,Flatten ,GlobalAveragePooling2D
from keras.applications import VGG16 ,InceptionV3
from keras.datasets import cifar100
import numpy as np
#함수형으로 만들기 

base_model = InceptionV3(weights = "imagenet", include_top=False)
# base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)

output1 = Dense(100, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output1)

# #1.
# for layer in base_model.layers: #레이어별로 한단씩 동결
#     layer.trainable = False
# Total params: 22,353,028
# Trainable params: 550,244
# Non-trainable params: 21,802,784
#2.
base_model.trainable = False
# Total params: 22,353,028
# Trainable params: 550,244
# Non-trainable params: 21,802,784
model.summary()

