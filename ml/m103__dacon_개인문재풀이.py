# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 

# #.1 데이터

# bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
#                 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
#                 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

# bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
#                 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
#                 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# plt.scatter(bream_length,bream_weight)
# plt.scatter(smelt_length,smelt_weight)
# plt.xlabel('length')
# plt.ylabel("weight")
# plt.show()

# weigth = bream_weight + smelt_weight
# length = bream_length + smelt_length

# fish_data = [[l,w] for l,w in zip(length,weigth)]

# print(fish_data)  

# fish_target = [1] * 35 + [0] * 14
# print(fish_target)

# #.2 모델구성
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# kn = KNeighborsClassifier(n_neighbors=17) 

# #.3 훈련
# kn.fit(fish_data,fish_target)

# kn.predict([[30,600]])

# #.4 평가, 예측
# # kn.score(fish_data, fish_target)
# results = kn.score(fish_data, fish_target)
# print('결과:', results)

# plt.scatter(bream_length,bream_weight)
# plt.scatter(smelt_length,smelt_weight)
# plt.scatter(30,600, marker='*', c='red') # matplotlib 색상코드
# plt.xlabel('length')
# plt.ylabel("weight")
# plt.show()

# for n in range(5,50):
#     kn.n_neighbors = n 
#     score = kn.score(fish_data,fish_target)
#     if score < 1:
#         print(n, score)
#         break

######################## 두번째 문제 !!################################################
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#.1 데이터

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0 ,9.8, 10.5,
                10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0,6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
fish_data = [[l,w] for l,w in zip(fish_length, fish_weight)]

fish_target = [1]*35 + [0]*14

print(fish_data)#35개의 데이터
# [[25.4, 242.0], [26.3, 290.0], [26.5, 340.0], [29.0, 363.0], [29.0, 430.0], [29.7, 450.0], [29.7, 500.0], [30.0, 390.0], [30.0, 450.0],
#  [30.7, 500.0], [31.0, 475.0], [31.0, 500.0], [31.5, 500.0], [32.0, 340.0], [32.0, 600.0], [32.0, 600.0], [33.0, 700.0], [33.0, 700.0],
#  [33.5, 610.0], [33.5, 650.0], [34.0, 575.0], [34.0, 685.0], [34.5, 620.0], [35.0, 680.0], [35.0, 700.0], [35.0, 725.0], [35.0, 720.0], 
#  [36.0, 714.0], [36.0, 850.0], [37.0, 1000.0], [38.5, 920.0], [38.5, 955.0], [39.5, 925.0], [41.0, 975.0], [41.0, 950.0], [9.8, 6.7], 
#  [10.5, 7.5], [10.6, 7.0], [11.0, 9.7], [11.2, 9.8],[11.3, 8.7], [11.8, 10.0], [11.8, 9.9], [12.0, 9.8], [12.2, 12.2], [12.4, 13.4], 
#  [13.0, 12.2], [14.3, 19.7], [15.0, 19.9]]

print(fish_target)#14개의 데이터
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
#  0, 0, 0, 0, 0, 0]

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)
print(input_arr.shape)  #(49, 2)

np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)


print(input_arr[[1,3]])


train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

print(input_arr[13],train_input[0])


test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

print(train_input.shape)    #(35, 2)
print(train_target.shape)   #(35,)
print(test_input.shape)     #(14, 2)
print(test_target.shape)    #(14,)



import matplotlib.pyplot as plt
plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(test_input[:,0],test_input[:,1])
plt.xlabel('length')
plt.ylabel('weigth')
plt.show()

#2. 모델 구성

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier,XGBRFRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, r2_score


xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier() #(verbose=False)#catboost vervose가 많음 ! 그래서 다른모델이랑 성능비교 시에는 주석처리

#voting 은 hard &soft가있음 #estimators= 두개이상은 리스트로 넣어줘야함
model = VotingClassifier(estimators=[('xg', xg), ('cat', cat),("lg", lg)], voting='soft') 


model =model.fit(train_input, train_target)

score = accuracy_score(test_input,test_target)

predicr = model.predict(test_input)

y_predict = model.predict(test_input)
print(model.score(test_input,test_target))


classifier = [cat,xg, lg,]

for model in classifier:  #model2는 모델이름 # 
    model.fit(train_input,train_target)
    y_predict = model.predict(test_input)
    score2 = accuracy_score(test_target,y_predict)
    class_name = model.__class__.__name__  #<모델이름 반환해줌 
    print("{0}정확도 : {1:.4f}".format(class_name, score2)) # f = format
    
print("보팅결과 : ", round(score,4 ))


print(predicr)
print(test_target)
print(score)