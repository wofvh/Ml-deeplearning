from unittest import result
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score , GridSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#1.데이터

path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)

x = train_set.drop(['count'], axis=1)     #drop 지울떄 사용함
x = train_set.drop(['count'], axis=1)     #drop 지울떄 사용함
print(x)
print(x.columns)
print(x.shape)       #(1459, 9)

y = train_set['count']

y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, shuffle= True,
                                                    random_state=72 )



n_splits = 5
Kfold = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=66)


# #2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier    #
from sklearn.tree import DecisionTreeClassifier       # 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor   # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 

Parameters= [
    {"n_estimators":[100,200], "max_depth":[6,12,14],'min_samples_leaf':[3, 10]},
    {"max_depth": [6, 82, 10 ,12],"min_samples_leaf" :[4, 16, 8,10],},
    {'min_samples_leaf':[3,15,71,10],"n_jobs":[14,20,9,12],"max_depth":[6, 18, 10 ,12]},
    {"min_samples_split":[2,3,15,10],"min_samples_split":[15,20,15,12],'min_samples_leaf':[12,25,7,10],},
    {'n_jobs':[-1,2,4],"max_depth":[6,11,12],"max_depth":[6, 8, 10 ,12]}
]

model = RandomizedSearchCV(RandomForestRegressor(),Parameters, cv =Kfold, verbose=1, refit=True, n_jobs=-1)
#컴파일 훈련
import time
start = time.time()

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ",model.best_params_)

print('best_score_:', model.best_score_)

print("model.score:", model.score(x_test,y_test))

#평가,예측\
end_time = time.time()
  
y_predict = model.predict(x_test)
print('accuracy_score:', r2_score(y_test, y_predict))
# print('accuracy', results[1])

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC:', r2_score(y_test, y_pred_best))
