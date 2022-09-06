import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Input,Dropout
from tensorflow.python.keras.models import sequential, Model
import tensorflow as tf
print(tf.__version__)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

print(x_train.shape ,y_train.shape,x_test.shape,y_test.shape) #(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#모델
def build_model(drop=0.5,optimizer='adam',activation='relu'):
    inputs = Input(shape=(28*28),name='input')
    x = Dense(256,activation=activation,name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(64,2,activation=activation,name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense()(x)
    x = Dense(32,activation=activation,name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name='outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer=optimizer,metrics=['acc'],loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [100,200,300,400,500]
    optimizers = ['rmsprop','adam','adadelta']
    dropout = np.linspace(0.1,0.5,5)
    activation = ['relu','elu','selu',"sifmoid","linear"]
    return{"batch_size":batches,"optimizer":optimizers,"drop":dropout,"activation":activation}

hyperparameters = create_hyperparameters()
print(hyperparameters)

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
