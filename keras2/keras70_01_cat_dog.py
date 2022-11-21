import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications import ResNet101V2
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAvgPool2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import time

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                validation_split=0.2)


test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'D:/study_data/_data/image/cat_dog/train_set',
    target_size = (50, 50),
    batch_size = 32,
    class_mode = 'binary',
    subset='training',
    shuffle = True)

xy_val = train_datagen.flow_from_directory(
    'D:/study_data/_data/image/cat_dog/train_set',
    target_size = (50, 50),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'validation')   

xy_test = test_datagen.flow_from_directory(
    'D:/study_data/_data/image/cat_dog/test_set',
    target_size= (50, 50),
    batch_size = 32,
    class_mode = 'binary')

#2. 모델
pre_train = ResNet101V2(weights='imagenet', include_top=False,
              input_shape=(50, 50, 3))
pre_train.trainable = True

model = Sequential()
model.add(pre_train)
model.add(GlobalAvgPool2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

start = time.time()
hist = model.fit(xy_train, epochs=100, steps_per_epoch=20, validation_data=xy_val, validation_steps=40, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 예측
loss = model.evaluate(xy_test)
print("걸린 시간 : ", round(end, 2))
print("loss, acc : ", loss)

'''
걸린 시간 :  165.28
loss, acc :  [0.4821479618549347, 0.8042511343955994]
'''

# 걸린 시간 :  328.9
# loss, acc :  [3.810443013207987e-05, 1.0]