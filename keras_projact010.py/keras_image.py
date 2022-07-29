import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

data = pd.read_csv('C:/Users/USER/Documents/test.csv',index_col=0) # data 불러오기
data.rename(columns={'0':'image_path'},inplace=True) # column name 변경

def cvt_pathname(path):
    path = path.replace('\\','/') #=> '\' => '/' 로 이름 정렬해줌
    return path

data['image_path'] = data['image_path'].apply(cvt_pathname)  # 'image_path' apply함수 이용해서 정렬

## Label 정리하기

def get_label(path):
    label = path.split('/')[-2] # '/' 로 split 해서 list형식으로 바꿔주고 폴더명으로 label 가져옴
    return label
data['label'] = data['image_path'].apply(get_label) #마찬가지 image_path 이용해서 위의 함수로 label 뽑아냄
print(len(data['label'].unique())) # 갯수확인
data['label'].value_counts().plot(kind='barh')

label_encoder = LabelEncoder() # sckit-learn label encoder 활용 string 형식 label 숫자로 변환
data['label'] = label_encoder.fit_transform(np.array(data['label'])) # fit_transform 이용 문자열 label 수치형(int)으로 변환
# label_encoder.inverse_transform(data['label'])
# data['label'].value_counts().plot(kind='barh')