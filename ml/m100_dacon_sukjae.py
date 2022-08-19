import pandas as pd
import random
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier,XGBRegressor  

path = './_data/prodacon/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

train_df = pd.read_csv(path + 'train.csv')
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
train = np.array(train_df)

# print("=============================상관계수 히트 맵==============")
# print(train_df.corr())                    # 상관관계를 확인.  
# import matplotlib.pyplot as plt 
# import seaborn as sns
# sns.set(font_scale=0.3)
# sns.heatmap(data=train_df.corr(),square=True, annot=True, cbar=True) 
# plt.show()

precent = [0.20,0.40,0.60,0.80]

print(train_df.describe(percentiles=precent))
# print(train_df.info())  
# print(train_df.columns.values)
# print(train_df.isnull().sum())

#  X_07, X_08, X_09
 
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

cols = ["X_10","X_11"]
train_x[cols] = train_x[cols].replace(0, np.nan)

# MICE 결측치 보간

imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman')


train_x = pd.DataFrame(imp.fit_transform(train_x))
print(train_x)
######################모델######################################
from sklearn.linear_model import LogisticRegression
# model = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
# 0.03932714821910016

# model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(train_x, train_y)
# 0.28798862985210744 best 

# model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=0.75, colsample_bytree = 1, max_depth=3) ).fit(train_x, train_y)
# 0.098387698230517

model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=0.75, colsample_bytree = 1, max_depth=3) ).fit(train_x, train_y)
# 0.098387698230517

# model = XGBRegressor().fit(train_x, train_y)
# 0.4177584378415335

print('Done.')
######################모델######################################

preds = model.predict(test_x)
print(preds)
print(model.score(train_x, train_y))
print('Done.')


submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv(path + 'submmit.csv', index=False)