from winreg import FlushKey
import numpy as np
import pandas as pd

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8, np.nan],
                     [2, 4, 6, 8 ,10],
                     [np.nan, 4, np.nan, 8,np.nan]])

print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)
print(data.shape)

#결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1.결측치 삭제
print('=============결측치 삭제=================')
print(data.dropna())
print(data.dropna(axis=1))  

#2-1. 특정값 - 평균 
means = data.mean()
print('평균:',means)
