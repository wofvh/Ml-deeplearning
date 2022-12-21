import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Sequential 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "C:\study\_data\kaggle_titanic/"

train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")


print(train.head())
print(test.head())
print(train.shape)  #(891, 12)
print(test.shape)   #(418, 11)

print(train.info())
print(train.describe()) #컬럼 평균치
print("=============================")

print(train['Survived'].value_counts()) #죽은사람 549 #산사람 342

train['Survived'].value_counts().plot.bar()


print(train['Survived'].value_counts().plot.bar())
plt.show()

