from bitarray import test
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from pyrsistent import v
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
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
# ########테스트 데이터는 증폭 안함#####
test_datagen = ImageDataGenerator(
    rescale=1./255
)

x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)

print(x_train1)
'''
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
x_train1 = train_datagen.flow_from_directory(
    'D:/test/choiminsik',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True, 
)
xy_test = test_datagen.flow_from_directory(
    'D:/study_data/_data/image/cat_dog/test_set/test_set',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True,
    #Found 120 images belonging to 2 classes 0~1로 데이터가 됬다
)

# <keras.preprocessing.image.DirectoryIterator object at 0x000001F0E08D7A90>

# print(xy_train[0])            #마지막 배치 
# print(xy_train[0][0])  
# print(xy_train[0][1])     #(5, 150, 150, 1) 

print(xy_train)  #(5, 200, 200, 3) (5,)
print(xy_test)   #(5, 200, 200, 3) (5,)


print(xy_train[0][0].shape, xy_train[0][1].shape)#(160, 150, 150, 1) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape) #(120, 150, 150, 1) (120,)

np.save('d:/study_data/_save/_npy/cat_dog_1_train_x.npy',arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/cat_dog_1_train_y.npy',arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/cat_dog_1_test_x.npy',arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/cat_dog_1_test_y.npy',arr=xy_test[0][1])

# print(type(xy_train))     #반복자 DirectoryIterator
# print(type(xy_train[0]))  #<class 'tuple'> 수정할수없는 Lsit
# print(type(xy_train[0][0]))  #<class 'numpy.ndarray'>
# print(type(xy_train[0][1]))  #<class 'numpy.ndarray'>


#현대 5,200,200,1 짜리 데이터가 32 덩어러 



#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Conv2D , Flatten,MaxPool2D

model = Sequential()
model.add(Conv2D(35,(2,2),input_shape = (200,200,3), activation='relu'))
model.add(Conv2D(64,(3,3),activation= 'relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(18,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

#3.컴파일 훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#model.fit(cy_train[0][0], xy_train[0][1])# 배치를 최대로 자으면 이것도 가능 
hist = model.fit_generator(xy_train ,epochs=2,steps_per_epoch=32,
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


########### MEDIAPIPE##################
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                print(face_landmarks)
                mp_drawing.draw_landmarks(
                  image=image,
                  landmark_list=face_landmarks,
                  connections=mp_face_mesh.FACE_CONNECTIONS,
                  landmark_drawing_spec=drawing_spec,
                  connection_drawing_spec=drawing_spec)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
'''