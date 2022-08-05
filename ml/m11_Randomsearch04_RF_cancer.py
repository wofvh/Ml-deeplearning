
from mmap import ACCESS_WRITE
from sre_parse import FLAGS
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.model_selection import KFold, cross_val_score , GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold

#
# 1.데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.2, shuffle= True,
                                                    random_state=1234 )
n_splits = 5 #다섯 번씩 모든 데이터를 훈련해준다고 지정 

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

Parameters= [
    {"n_estimators":[100,200], "max_depth":[6,12,14],'min_samples_leaf':[3, 10]},
    {"max_depth": [6, 82, 10 ,12],"min_samples_leaf" :[4, 16, 8,10],},
    {'min_samples_leaf':[3,15,71,10],"n_jobs":[14,20,9,12],"max_depth":[6, 18, 10 ,12]},
    {"min_samples_split":[2,3,15,10],"min_samples_split":[15,20,15,12],'min_samples_leaf':[12,25,7,10],},
    {'n_jobs':[-1,2,4],"max_depth":[6,11,12],"max_depth":[6, 8, 10 ,12]}
]

# #2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier    #
from sklearn.tree import DecisionTreeClassifier       # 
from sklearn.ensemble import RandomForestClassifier   # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 
# model = SVC(C=1, kernel="linear", degree=3)
model = RandomizedSearchCV(RandomForestClassifier(),Parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)  #n_jobs cpu 갯수사용 정의 예) 1 cup 1 -1 cup8 

#RandomizedSearchC 10 개만 빼서 훈련함 

#컴파일 훈련
import time
start = time.time()
model.fit(x_train, y_train)
print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ",model.best_params_)

print('best_score_:', model.best_score_)

print("model.score:", model.score(x_test,y_test))
end_time = time.time()
#평가,예측\
  
y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))
# print('accuracy', results[1])

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test, y_pred_best))

print("걸린시간:", round(end_time - start, 4))

# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=10, n_jobs=12)
# 최적의 파라미터 :  {'n_jobs': 12, 'min_samples_leaf': 10, 'max_depth': 6}
# best_score_: 0.9549407114624506
# model.score: 0.9342105263157895
# accuracy_score: 0.9342105263157895
# 최적 튠 ACC: 0.9342105263157895
# 걸린시간: 2.2737
