# import numpy as np 
# import pandas as pd
# from xgboost import XGBClassifier,XGBRFRegressor

# # train 데이터 불러오기

# class DaconTravel(object):
#     def __init__(self,model,train,test):
#         path = './_data/travel/'
       
#         train = pd.read_csv(path + 'train.csv',index_col=0)                       

#         test = pd.read_csv(path + 'test.csv',  index_col=0)

# class DaconTravel:
    
#     def __init__(self,humen,age, total):
#         self.humen = humen
#         self.age = age
#         self.total = total
#         pass
    
#     def join(self):
#         print(f' total{self.total},{self.humen}18,{self.age}18')
        
# travers = DaconTravel('여행객','나이','인원')
# travers.join()
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
import numpy as np
import pandas as pd
import tensorflow as tf

class Mymodel(object):
        
    def data(self):
        x = (np.array([1,2,3]))
        # y = (np.array([1,2,3]))
        # self.y = (np.array([1,2,3]))
        print(f'데이터{x}')
        
        
        
result = Mymodel()
result.data()



  
# result = Mymodel.predict([4])
# print('4의 예측값 : ',result )


class Person:
    def greeting(self):
        print('hello')
        
maria = Person()
maria.greeting()
