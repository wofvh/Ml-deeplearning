import numpy as np
import pandas as pd                               
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from tqdm import tqdm_notebook
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
#1. 데이터
path = './_data/travel/'
train_df = pd.read_csv(path + 'train.csv',                 
                        index_col=0)                       

test_df = pd.read_csv(path + 'test.csv',                                  
                       index_col=0)

# print(train_df,test_df,)  #(train>1955, 19) (test>2933, 18)


train_df.rename(columns={
'id' : '아이디'
,'Age' : '나이'
,'TypeofContact' : '탐색경로'
,'CityTier' : '도시등급'
,'DurationOfPitch' : '프리젠테이션기간'
,'Occupation' : '직업'
,'Gender' : '성별'
,'NumberOfPersonVisiting' : '여행인원'
,'NumberOfFollowups' : '후속조치수'
,'ProductPitched' : '제시상품'
,'PreferredPropertyStar' : '선호숙박등급'
,'MaritalStatus' : '결혼여부'
,'NumberOfTrips' : '연간여행횟수'
,'Passport' : '여권보유'
,'PitchSatisfactionScore' : '프레젠테이션만족도'
,'OwnCar' : '자동차보유'
,'NumberOfChildrenVisiting' : '미취학아동'
,'Designation' : '직급'
,'MonthlyIncome' : '월급여'
,'ProdTaken' : '신청여부'
}, inplace = True)
test_df.rename(columns = {
'id' : '아이디'
,'Age' : '나이'
,'TypeofContact' : '탐색경로'
,'CityTier' : '도시등급'
,'DurationOfPitch' : '프리젠테이션기간'
,'Occupation' : '직업'
,'Gender' : '성별'
,'NumberOfPersonVisiting' : '여행인원'
,'NumberOfFollowups' : '후속조치수'
,'ProductPitched' : '제시상품'
,'PreferredPropertyStar' : '선호숙박등급'
,'MaritalStatus' : '결혼여부'
,'NumberOfTrips' : '연간여행횟수'
,'Passport' : '여권보유'
,'PitchSatisfactionScore' : '프레젠테이션만족도'
,'OwnCar' : '자동차보유'
,'NumberOfChildrenVisiting' : '미취학아동'
,'Designation' : '직급'
,'MonthlyIncome' : '월급여'
,'ProdTaken' : '신청여부'
}, inplace = True)

print(train_df.select_dtypes(exclude=['object']).columns) #숫자만 있음
train_df.describe()


cols = train_df.select_dtypes(exclude=['object']).columns
fig, axes = plt.subplots(2, 7,  figsize=(24,10))
for i, col in enumerate(cols):
    if i <=6:
        k = i
        j  = 0
    else:
        j = 1
        k = i-7
    g = sns.distplot(train_df[train_df['신청여부']==0][col], hist=True, kde=False, rug=True, bins=20, label="미신청"   , ax = axes[j][k])
    h = sns.distplot(train_df[train_df['신청여부']==1][col], hist=True, kde=False, rug=True, bins=20, label="신청", ax = axes[j][k])
    axes[j][k].legend()
    plt.xlabel(col)
plt.show()
plt.close()