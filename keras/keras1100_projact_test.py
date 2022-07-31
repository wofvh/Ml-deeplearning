from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import math
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input 


## 데이터 제네레이터##
## 쓰는 이유 : 이미지를 한 덩어리 안에 집어넣으면 램이 터지기때문에 학습할때 쪼개서 들어갈 수 있도록 class형식으로 만들어서 넣어줌

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, df, image_size, mode='train', shuffle=True): # 생성자 : 배치사이즈, 데이터프레임, 모드(학습,검증) , 셔플(섞어서들어가기)
        self.batch_size = batch_size 
        self.mode = mode
        self.image_size = image_size
        self.shuffle = shuffle
        self.df = df
 
        if self.mode == 'train': # mode가 train일때 검증용 데이터 (지금은 fold == 1 ) 빼고 학습을 진행
            self.df = self.df[self.df['fold'] != 1]
        elif self.mode =='val': # mode 가 validation(검증) 일때는 얘만 따로 학습검증을함
            self.df = self.df[self.df['fold'] == 1]

        self.on_epoch_end()


    def __len__(self): # 길이정의 (전체 dataset 길이에서 batch사이즈만큼 나눠줌)
        return math.ceil(len(self.df) / self.batch_size)


    def __getitem__(self, idx): # 실질적으로 빠지는 부분 
        strt = idx * self.batch_size 
        fin = (idx+1) * self.batch_size
        data = self.df.iloc[strt:fin] # ex) idx가 0 이고 배치사이즈 64 일 경우  dataframe의 [0:64] 그다음은 [64:128]로 찢어서 나갈 수 있게 iterator(반복자) 개념 필요
        batch_x, batch_y = self.get_data(data) # 아래 정의된 get data에서 데이터 받아옴

        return np.array(batch_x), np.array(batch_y)


    def get_data(self, data):
        batch_x = []
        batch_y = []

        for _ , r in data.iterrows(): # data iterrows 로 한개씩 훑고 지나감(이미지를 불러오기 위함) 연습필요 당장 이해 힘듬
            file_path = r['image_path'] 
            
            image = cv2.imread(file_path) #이미지 불러오고
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cvtColor 할거임

            image = cv2.resize(image , (self.image_size , self.image_size)) # model input size로 resize 해주기
            image = image / 255. # 정규화 0~1

            label = int(r['label']) # label로 지정된 int형 (정답지 : ground Truth)

            batch_x.append(image) # 앞선 리스트에 정의
            batch_y.append(label)

        return batch_x, batch_y # return => __getitem__

    def on_epoch_end(self):

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

from sklearn.model_selection import StratifiedKFold
import pandas as pd

data = pd.read_csv('./_data/project/cls_data.csv',index_col=0)
skf = StratifiedKFold(n_splits=8, shuffle = True, random_state=42)

data['fold'] = -1

for idx, (t, v) in enumerate(skf.split(data, data['label']), 1):
    data.loc[v, 'fold'] = idx


train_gen = DataGenerator(batch_size=4, df = data, mode = 'train', image_size = 128)
valid_gen = DataGenerator(batch_size=4, df = data, mode = 'valid', image_size = 128)

print(train_gen[0][0].shape)  #(4, 128, 128, 3)
print(train_gen[0][1].shape)  #(4,)
print(valid_gen[0][0].shape)  #(4, 128, 128, 3)
print(valid_gen[0][1].shape)  #(4,)


#모델구성 

model = Sequential()
model.add(Conv2D(35,(2,2),input_shape = (128,128,3), activation='relu'))
model.add(Conv2D(64,(3,3),activation= 'relu'))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(30,activation='softmax'))
model.summary()

print(train_gen)


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_gen , validation_data = valid_gen , epochs=30)


conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


##################################

from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import math
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input 
from bitarray import test
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.model_selection import train_test_split

#############이미지 수치화 or 증폭가능############## 
train_datagen = ImageDataGenerator(
    rescale=1./255,  # rescale다른 처리 전에 데이터를 곱할 값입니다.1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다
horizontal_flip=True, # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다
vertical_flip=True,  #수직방향으로 뒤집기를 한다
width_shift_range=0.1, # width_shift그림 을 height_shift수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
height_shift_range=0.1, #지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동시킨다. 예를 들어 0.1이고 전체 높이가 100이면, 10픽셀 내외로 상하 이동시킨다. 
rotation_range=5,   # rotation_range사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
zoom_range=1.2, 
shear_range=0.7, 
fill_mode='nearest'
)

# rotation_range사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
# width_shift그림 을 height_shift수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
# rescale다른 처리 전에 데이터를 곱할 값입니다. 원본 이미지는 0-255의 RGB 계수로 구성되지만 이러한 값은 모델이 처리하기에는 너무 높기 때문에(주어진 일반적인 학습률) 1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다. 요인.
# shear_range무작위로 전단 변환 을 적용하기 위한 것입니다.
# zoom_range내부 사진을 무작위로 확대하기 위한 것입니다.
# horizontal_flip이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다(예: 실제 사진).
# fill_mode회전 또는 너비/높이 이동 후에 나타날 수 있는 새로 생성된 픽셀을 채우는 데 사용되는 전략입니다.


path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv',)#(10886, 11)

test_set = pd.read_csv(path + 'test.csv',)  #(6493, 8)

xy = train_datagen.flow_from_directory(
    'D:\study_data\_data\image\men_women\\adam',
    target_size=(150,150),
    batch_size=10000,
    class_mode='binary',
    shuffle = True,
    #Found 160 images belonging to 2 classes 160 데이터가 0~1로 데이터가 됬다
    #타겟싸이즈 맞춰야함 
)

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Conv2D , Flatten,MaxPool2D

model = Sequential()
model.add(Conv2D(35,(2,2),input_shape = (150,150,3), activation='relu'))
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
hist = model.fit_generator(xy ,epochs=2,steps_per_epoch=32,
                                            #전체데이터/batch = 160/5 = 32
                    validation_data =xy, 
                    validation_steps=4)
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']