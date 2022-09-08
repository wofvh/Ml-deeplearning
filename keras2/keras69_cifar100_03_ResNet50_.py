from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications import ResNet50
from keras.applications.vgg19 import preprocess_input, decode_predictions
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

resnet50 = ResNet50(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
resnet50.trainable = True

model = Sequential()
model.add(resnet50)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print('loss, acc ', loss)


'''
걸린 시간 :  512.3
loss, acc  [4.60548210144043, 0.009999999776482582]
'''

'''
preprocess_input
걸린 시간 :  3258.96
loss, acc  [2.370940923690796, 0.3935999870300293]
'''

# 걸린 시간 :  4121.2
# loss, acc  [2.4682798385620117, 0.3824999928474426]