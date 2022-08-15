import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer # 이터러블 입력시 사용하는 모듈 추가
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel




#1. 데이터
path = './_data/travel/'
train = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

test = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# 결측치를 처리하는 함수를 작성.
# drop_col = ['NumberOfChildrenVisiting','TypeofContact','OwnCar','NumberOfPersonVisiting'] # 컬럼 삭제하기 위한 리스트 생성
# train = train.drop(drop_col, axis=1) # axis=1 : 세로, axis=0 : 가로
# test = test.drop(drop_col, axis=1) # 결측치 처리하기 위한 함수 실행

def handle_na(data):
    temp = data.copy()
    for col, dtype in temp.dtypes.items():
        if dtype == 'object':
            # 문자형 칼럼의 경우 'Unknown'
            value = 'Unknown'
        elif dtype == int or dtype == float:
            # 수치형 칼럼의 경우 0
            value = 0
        temp.loc[:,col] = temp[col].fillna(value)
    return temp

train_nona = handle_na(train)

# 결측치 처리가 잘 되었는지 확인해 줍니다.
train_nona.isna().sum()

print(train_nona.isna().sum())
object_columns = train_nona.columns[train_nona.dtypes == 'object']
print('object 칼럼 : ', list(object_columns))

# 해당 칼럼만 보아서 봅시다
train_nona[object_columns]

# LabelEncoder를 준비해줍니다.
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# LabelEcoder는 학습하는 과정을 필요로 합니다.
# encoder.fit(train_nona['TypeofContact'])

#학습된 encoder를 사용하여 문자형 변수를 숫자로 변환해줍니다.
# encoder.transform(train_nona['TypeofContact'])
# print(train_nona['TypeofContact'])

train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결과를 확인합니다.
print(train_enc)
# 결측치 처리
test = handle_na(test)

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    
    # test 데이터를 이용해 encoder를 학습하는 것은 Data Leakage 입니다! 조심!
    encoder.fit(train_nona[o_col])
    
    # test 데이터는 오로지 transform 에서만 사용되어야 합니다.
    test[o_col] = encoder.transform(test[o_col])

# 결과를 확인
print(test)


# 모델 선언
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
# model = XGBClassifier()
model = XGBClassifier()

# # 분석할 의미가 없는 칼럼을 제거합니다.
# train = train_enc.drop(columns=['TypeofContact','Occupation'])
# test = test.drop(columns=['TypeofContact','Occupation'])


# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x = train_enc.drop(columns=['ProdTaken'])
y = train_enc[['ProdTaken']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.head())
print(y_train.head())




#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train, y_train)  # **fit_params
end = time.time()- start
#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result) # model.score :  1.0

y_predict = model.predict(x_test)
print('accuracy_score :',accuracy_score(y_test,y_predict))



pred = model.predict(test)
y_summit = [1 if x > 0.5 else 0 for x in pred]

submission_set = pd.read_csv(path + 'sample_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

submission_set['ProdTaken'] = y_summit

submission_set.to_csv(path + 'sample_submission_cat_drp.csv', index = True)