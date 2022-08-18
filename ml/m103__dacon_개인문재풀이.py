import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#.1 데이터

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length,bream_weight)
plt.scatter(smelt_length,smelt_weight)
plt.xlabel('length')
plt.ylabel("weight")
plt.show()

weigth = bream_weight + smelt_weight
length = bream_length + smelt_length

fish_data = [[l,w] for l,w in zip(length,weigth)]

# print(fish_data)  

fish_target = [1] * 35 + [0] * 14
# print(fish_target)

#.2 모델구성
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

kn = KNeighborsClassifier(n_neighbors=17) 

#.3 훈련
kn.fit(fish_data,fish_target)

kn.predict([[30,600]])

#.4 평가, 예측
# kn.score(fish_data, fish_target)
results = kn.score(fish_data, fish_target)
print('결과:', results)

# plt.scatter(bream_length,bream_weight)
# plt.scatter(smelt_length,smelt_weight)
# plt.scatter(30,600, marker='*', c='red') # matplotlib 색상코드
# plt.xlabel('length')
# plt.ylabel("weight")
# plt.show()

for n in range(5,50):
    kn.n_neighbors = n 
    score = kn.score(fish_data,fish_target)
    if score < 1:
        print(n, score)
        break