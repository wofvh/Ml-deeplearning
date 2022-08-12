# 문제 풀기 
# from tensorflow.python.keras.models import Sequential,load_model
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
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)


submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
                       
print(test_set)          #[2933 rows x 18 columns]
print(test_set.shape)    #(2933, 18)

print(train_set)          #[1955 rows x 19 columns]
print(train_set.shape)    #(1955, 19)      


print(train_set.isna().sum())
#train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_set[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

def handle_na(data):
    temp = data.copy()
    for col, dtype in temp.dtypes.items():
        if dtype == 'object':
            #문자형 컬럼의 경우 "unknow"k 를 채워줌
            value ='unknown'
        elif dtype == int or dtype == float:
            #수치형 컬럼의 경우0을 채워줌
            value = 0
        temp.loc[:,col] = temp[col].fillna(value)
    return temp

train_set = handle_na(train_set)

print(train_set.isna().sum())

e = LabelEncoder()
y = le.fit_transform(y)
print(y.shape)

# pca = PCA(n_components=20)   #54 >10
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x,y)
x = lda.transform(x)
print(x.shape)
print(y.shape)



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
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# parameters = [
#     {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
#      'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
#     {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
#      'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
#     ]     
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import make_pipeline

# model= SVC()
model = make_pipeline(MinMaxScaler(),RandomForestRegressor())

#훈련
model.fit(x_train,y_train)

#평가예측
result = model.score(x_test,y_test)

print("model.score:", result)
'''