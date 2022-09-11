#### DenseNet121
import numpy as np 
from keras.models import 
from keres.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.applications import Dense,Flatten,GlobalAveragePooling2D

from keras.dataset import difar100

den121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3))

model = Sequential()
model.add(den121)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.sunmmary()

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience=5, mode='auto', verbose=1, restore_best_weights=True)
acc = accuracy_score(y_test, y_pred)

print("acc : ", acc)
