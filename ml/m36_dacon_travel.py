import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import time
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_data/travel/'
train = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train)
print() #(1459, 10)

test = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)

# print(train.head())
# print(train.info())
# print(train.isnull().sum())

# 결측치 컨텍빼고 중간값으로 대체함
train['Age'].fillna(train['Age'].median(), inplace=True)
train['TypeofContact'].fillna('N', inplace=True)
train['DurationOfPitch'].fillna(train['DurationOfPitch'].median(), inplace=True)
train['NumberOfFollowups'].fillna(train['NumberOfFollowups'].median(), inplace=True)
train['PreferredPropertyStar'].fillna(train['PreferredPropertyStar'].median(), inplace=True)
train['NumberOfTrips'].fillna(train['NumberOfTrips'].median(), inplace=True)
train['NumberOfChildrenVisiting'].fillna(train['NumberOfChildrenVisiting'].median(), inplace=True)
train['MonthlyIncome'].fillna(train['MonthlyIncome'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)
test['TypeofContact'].fillna('N', inplace=True)
test['DurationOfPitch'].fillna(test['DurationOfPitch'].median(), inplace=True)
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].median(), inplace=True)
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].median(), inplace=True)
test['NumberOfTrips'].fillna(test['NumberOfTrips'].median(), inplace=True)
test['NumberOfChildrenVisiting'].fillna(test['NumberOfChildrenVisiting'].median(), inplace=True)
test['MonthlyIncome'].fillna(test['MonthlyIncome'].median(), inplace=True)
# print(train.isnull().sum())

# object타입 라벨인코딩

x = train.drop('ProdTaken', axis=1)
y = train['ProdTaken']
x = np.array(x)
y = np.array(y)
y = y.reshape(-1, 1)
test = np.array(test)
print(x.shape, y.shape) # (1955, 19) (1955, 1)

import matplotlib.pyplot as plt
import math


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3-quartile_1 # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def outliers_printer(dataset):
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        
    plt.show()

outliers_printer(x)

# PCA 반복문
# for i in range(x.shape[1]):
#     pca = PCA(n_components=i+1)
#     x2 = pca.fit_transform(x)
#     x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=123, shuffle=True)
#     model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
#     model.fit(x_train, y_train)
#     results = model.score(x_test, y_test)
#     print(i+1, '의 결과: ', results)


parameters = {
            'n_estimators':[100,200,300,400,500],
            'learning_rate':[0.1,0.2,0.3,0.5,1,0.01,0.001],
            'max_depth':[None,2,3,4,5,6,7,8,9,10],
            'gamma':[0,1,2,3,4,5,7,10,100],
            'min_child_weight':[0,0.1,0.001,0.5,1,5,10,100],
            'subsample':[0,0.1,0.2,0.3,0.5,0.7,1],
            'reg_alpha':[0,0.1,0.01,0.001,1,2,10],
            'reg_lambda':[0,0.1,0.01,0.001,1,2,10],
              } 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)
# print(np.unique(y_train, return_counts=True))

# 2. 모델
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
model = make_pipeline(MinMaxScaler(), HalvingRandomSearchCV(xgb, parameters, cv=5, n_jobs=-1, verbose=2))

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)

# 5. 제출 준비
y_submit = model.predict(test)
submission['ProdTaken'] = y_submit

submission.to_csv(path + 'submission.csv', index = True)
