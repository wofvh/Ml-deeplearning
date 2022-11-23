from tabnanny import verbose
import numpy as np
from torch import dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2. model
optimizer='adam'
drop=0.2
activation='relu'

inputs = Input(shape=(28,28,1), name='input')
x = Conv2D(64,(2,2),padding='valid', 
           activation=activation, name='hidden1')(inputs) #27.27.128
x = Dropout(drop)(x)

x = MaxPooling2D()(x)

x = Conv2D(32,(3,3),padding='valid',
           activation=activation, name='hidden3')(x)      #27.27.128
x = Dropout(drop)(x)
x = GlobalAveragePooling2D()(x)     #25*25*32 =20000

x = Dense(100, activation=activation,name ="hidden4" )(x)      #27.27.128
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(optimizer=optimizer,metrics = ['acc'],
              loss='sparse_categorical_crossentropy')

import time
start = time.time()
model.fit(x_train,y_train, epochs=5, validation_split=0.4,batch_size=128)
end = time.time()

loss, acc = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

# print(y_predict[:10])
y_predict = np.argmax(model.predict(x_test), axis=-1)

print('걸린시간 : ', end-start)
print('acc : ', accuracy_score(y_test,y_predict))

# 걸린시간 :  14.816811084747314
# acc :  0.8242