import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.svm import LinearSVC,LinearSVR
from sklearn import datasets                 #분류 & 회기
from sklearn.linear_model import LogisticRegression,LinearRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor    #
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor     # 
from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor  # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

#1.데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

kfold = KFold(n_splits= n_splits, )



# (455,)
# (455, 30)

#2. 모델구성\
allAgorithms = all_estimators(type_filter='regressor') #회기모델
print('allAgorithms:', allAgorithms )
print('모델의 갯수:',len(allAgorithms)) #41
