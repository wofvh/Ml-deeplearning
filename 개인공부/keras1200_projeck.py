from bitarray import test
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets, utils
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import cv2
import math
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from sklearn.model_selection import train_test_split

#############이미지 수치화 or 증폭가능############## 
train_datagen = ImageDataGenerator(
    rescale=1./255,  # rescale다른 처리 전에 데이터를 곱할 값입니다.1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다
    horizontal_flip=True, # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다
    vertical_flip=True,  #수직방향으로 뒤집기를 한다
    width_shift_range=0.1, # width_shift그림 을 height_shift수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    height_shift_range=0.1, #지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동시킨다. 예를 들어 0.1이고 전체 높이가 100이면, 10픽셀 내외로 상하 이동시킨다. 
    rotation_range=5,   # rotation_range사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
    zoom_range=1.2,   # zoom_range내부 사진을 무작위로 확대하기 위한 것입니다.
    shear_range=0.7,  # shear_range무작위로 전단 변환 을 적용하기 위한 것입니다.
    fill_mode='nearest'
)
# rotation_range사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
# width_shift그림 을 height_shift수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
# rescale다른 처리 전에 데이터를 곱할 값입니다. 원본 이미지는 0-255의 RGB 계수로 구성되지만 이러한 값은 모델이 처리하기에는 너무 높기 때문에(주어진 일반적인 학습률) 1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다. 요인.
# shear_range무작위로 전단 변환 을 적용하기 위한 것입니다.
# zoom_range내부 사진을 무작위로 확대하기 위한 것입니다.
# horizontal_flip이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다(예: 실제 사진).
# fill_mode회전 또는 너비/높이 이동 후에 나타날 수 있는 새로 생성된 픽셀을 채우는 데 사용되는 전략입니다.

########테스트 데이터는 증폭 안함#####
test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'D:/study_data/test',
    target_size=(100,100),
    batch_size=5000,
    class_mode='categorical',  
    shuffle = True,
    #Found 160 images belonging to 2 classes 160 데이터가 0~1로 데이터가 됬다
    #타겟싸이즈 맞춰야함 
)


x = xy_train[0][0]
y = xy_train[0][1]


x_train, x_test, y_train, y_test = train_test_split(
     x, y, train_size=0.8, shuffle=True
    )


print(x_train.shape) #(21, 100, 100, 3)


#print(x.shape, y.shape)

#batch_size = 64

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Conv2D , Flatten,MaxPooling2D


model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(2, 2), padding='same', input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(512,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(30, activation='softmax'))
model.summary()

#3.컴파일 훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#model.fit(cy_train[0][0], xy_train[0][1])# 배치를 최대로 자으면 이것도 가능 
hist = model.fit_generator(xy_train ,epochs=30,steps_per_epoch=32,
                                            #전체데이터/batch = 160/5 = 32
                    validation_data =xy_test, 
                    validation_steps=4)
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ' ,loss[-1])
print('val_loss : ' ,val_loss[-1])
print('accuracy : ' ,accuracy[-1])
print('val_accuracy : ' ,val_accuracy[-1])


import matplotlib.pyplot as plt
matplotlib.rcParams
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('show') #맥플러립 한글 깨짐 현상 알아서 해결해라 
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend()
plt.show()

##########################################################################################################
# x_train = np.load('D:\study_data\_save\_npy\_train_x10.npy')
# y_train = np.load('D:\study_data\_save\_npy\_train_y10.npy')
# x_test = np.load('D:\study_data\_save\_npy\_test_x10.npy')
# y_test = np.load('D:\study_data\_save\_npy\_test_y10.npy')

#2. 모델 

# model = Sequential()
# model.add(Conv2D(input_shape=(100, 100, 1), kernel_size=(3, 3), filters=32, padding='same', activation='relu'))
# model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'))
# model.add(MaxPool2D((2, 2)))
# model.add(Dropout(0.5))

# model.add(Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'))
# model.add(Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'))
# model.add(MaxPool2D((2, 2)))
# model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(21,activation='softmax'))
# # model.summary()


model.load_weights("D:\study_data\_save\keras60_project4.h5")
start_time = time.time()
#3. 컴파일,훈련
# filepath = './_test/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k24_', date, '_', filename])
#                     )
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# hist = model.fit(x_train,y_train,epochs=50,verbose=2,
#                  validation_split=0.25,
#                  callbacks=[earlyStopping]
#                  ,batch_size=500)
# model.save_weights("D:\study_data\_save\keras60_project4.h5")
# model.save_weights("./_save/keras23_5_save_weights1.h5")

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
# print('loss :', loss)
# end_time = time.time()-start_time
print("걸린 시간 :",end_time)
x_data = np.load('D:\study_data\_save\_npy\_train_x12.npy')
y_predict = model.predict(x_data)
y_predict = np.argmax(y_predict,axis=1)
print('y_predict :',y_predict) 
from random import *

# is_bal = df['Genre'] == '발라드'

# 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
# bal = df[is_bal]
# i = randrange(40)  # 0부터 39 사이의 임의의 정수
# print(i)
# bal = '{} - {}'.format(bal['title'][i],bal['artist'][i])
# 결과를 출력합니다.
if y_predict[0]   ==   1  : print('분노한 표정-추천 노래 :',bal)
elif y_predict[0] ==   2  : print('혐오하는 표정-추천 노래 :',bal)
elif y_predict[0] ==   3  : print('공포스러워하는 표정-추천 노래 :',bal)
elif y_predict[0] ==   4  : print('행복해하는 표정-추천 노래 :',bal)
elif y_predict[0] ==   5  : print('무표정-추천 노래 :',bal)
elif y_predict[0] ==   6  : print('슬픈 표정-추천 노래 :',bal)
elif y_predict[0] ==   7  : print('놀라워하는 표정-추천 노래 :',bal)  
elif y_predict[0] ==   8  : print('불안한 표정-추천 노래 :',bal)
elif y_predict[0] ==   9  : print('감동받은 표정-추천 노래 :',bal)
elif y_predict[0] ==   10 : print('지루한 표정-추천 노래 :',bal)
elif y_predict[0] ==   11 : print('의기양양한 표정-추천 노래 :',bal)
elif y_predict[0] ==   12 : print('실망한 표정-추천 노래 :',bal)
elif y_predict[0] ==   13 : print('의심하는 표정-추천 노래 :',bal)  
elif y_predict[0] ==   14 : print('흥미로운 표정-추천 노래 :',bal)
elif y_predict[0] ==   15 : print('죄책감 표정-추천 노래 :',bal)
elif y_predict[0] ==   16 : print('질투 표정-추천 노래 :',bal)
elif y_predict[0] ==   17 : print('외로운 표정-추천 노래 :',bal)
elif y_predict[0] ==   18 : print('만족한 표정-추천 노래 :',bal)
elif y_predict[0] ==   19 : print('진지한 표정-추천 노래 :',bal)  
elif y_predict[0] ==   20 : print('억울한 표정-추천 노래 :',bal)
elif y_predict[0] ==   21 : print('승리한 표정-추천 노래 :',bal)  
'''