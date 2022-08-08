from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import matplotlib
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
                       index_col=0)
                       
print(test_set)
print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set['count']

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor #공부하자 

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = [
    {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
     'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
    {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
     'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
    ]     
#2. 모델 구성
model = HalvingRandomSearchCV(RandomForestRegressor(),parameters,cv=kfold,verbose=1,
                     refit=True,n_jobs=-1) 

#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train,y_train) 
end = time.time()- start

print("최적의 매개변수 :",model.best_estimator_)
print("최적의 파라미터 :",model.best_params_)
print("best_score :",model.best_score_)
print("model_score :",model.score(x_test,y_test))
y_predict = model.predict(x_test)
print('accuracy_score :',r2_score(y_test,y_predict))
y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',r2_score(y_test,y_predict))
print("걸린 시간 :",round(end,2),"초")  
#==================gridsearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=3, n_estimators=200,
#                       n_jobs=-1)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 200, 'n_jobs': -1}  
# best_score : 0.7518449572796932
# model_score : 0.787842022844681
# accuracy_score : 0.787842022844681
# 최적 튠  ACC : 0.787842022844681
# 걸린 시간 : 26.71 초
#==================randomsearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=3, min_samples_split=3,
#                       n_jobs=-1)
# 최적의 파라미터 : {'n_jobs': -1, 'n_estimators': 100, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_depth': 8}  
# best_score : 0.7495962567080064
# model_score : 0.7897321106506551
# accuracy_score : 0.7897321106506551
# 최적 튠  ACC : 0.7897321106506551
# 걸린 시간 : 5.47 초
#==================HalvingGridSearchCV    
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=7, min_samples_split=7,
#                       n_estimators=400, n_jobs=4)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 7, 'min_samples_split': 7, 'n_estimators': 400, 'n_jobs': 4}   
# best_score : 0.7384703595113962
# model_score : 0.7788312581828459
# accuracy_score : 0.7788312581828459
# 최적 튠  ACC : 0.7788312581828459
# 걸린 시간 : 29.87 초   
#==================HalvingrandomSearchCV   