# 증폭후 저장한 데이터를 불러와서 
# 완성및 성능 비교 

import select
from selectors import SelectSelector
from sklearn.datasets import load_breast_cancer , load_diabetes , load_iris ,fetch_california_housing,load_breast_cancer, load_wine,fetch_covtype
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(178, 13) (178,)

le = LabelEncoder()
