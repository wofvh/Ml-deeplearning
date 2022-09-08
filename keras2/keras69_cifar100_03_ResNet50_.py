from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG19
from keras.callbacks import EarlyStopping
import numpy as np
from keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import preprocess_input




