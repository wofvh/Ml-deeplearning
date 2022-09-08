from keras.applications import VGG16, VGG19, Xception, ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, 
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2 ,MobileNetV3Large, MobileNetV3Small
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobilefrom
from keras.applications import EfficientNetB0, EfficientNetB1,EfficientNetB7
from keras.applications import Xception

model = VGG16()
model = VGG19()

model.trainable = False
print("=================================")
print("모델명:", model.name)
print("모델의 가중치 개수:", len(model.weights))
print("모델의 훈련 가능한 가중치 개수:", len(model.trainable_weights))