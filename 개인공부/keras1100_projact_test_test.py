import tensorflow as tf
from mimetypes import init
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import math
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold

# Define Generator
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

            # label = r.iloc[:,2:-1] # label로 지정된 int형 (정답지 : ground Truth)
            label = r['label']
            batch_x.append(image) # 앞선 리스트에 정의
            batch_y.append(label)

        return batch_x, batch_y # return => __getitem__

    def on_epoch_end(self):

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

#################################################################################
################################# PATH 보완 #####################################
#################################################################################

data = pd.read_csv('./_data/project/cls_data.csv', index_col=0)
# data['image_path'] = data['image_path'].apply(cp)
# Fold
skf = StratifiedKFold(n_splits=8, shuffle = True, random_state=42)
data['fold'] = -1


for idx, (t, v) in enumerate(skf.split(data, data['label']), 1):
    data.loc[v, 'fold'] = idx
    
train_gen = DataGenerator(batch_size=16, df = data, mode = 'train', image_size = 94)
valid_gen = DataGenerator(batch_size=16, df = data, mode = 'valid', image_size = 94)
    
    
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(3, 3), padding='same', input_shape=(94,94,3), activation='relu'))
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
# model.summary()




#3. 컴파일, 훈련\
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_gen , validation_data = valid_gen , epochs=30)

