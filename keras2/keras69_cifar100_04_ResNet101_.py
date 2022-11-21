from threading import activeCount
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications import ResNet50,ResNet101
from keras.applications.vgg19 import preprocess_input, decode_predictions
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

X_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3))
resnet101._trainable = False

model = Sequential()
model.add(resnet101)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='sodftmax'))

model.compile