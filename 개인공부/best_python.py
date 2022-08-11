from tracemalloc import start
from turtle import shape
from unittest import result
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,fetch_covtype,load_digits
from sklearn.datasets import load_breast_cancer,load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
print('xgboost:', xg.__version__)

datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape)  #(150, 4)  
print(y.shape)

le = LabelEncoder()
y = le.fit_transform(y)
print(y.shape)

datainfo= ["id,Age,TypeofContact,CityTier,DurationOfPitch,Occupation,Gender,NumberOfPersonVisiting,\
    NumberOfFollowups,ProductPitched,PreferredPropertyStar,MaritalStatus,NumberOfTrips,Passport,\
    PitchSatisfactionScore,OwnCar,NumberOfChildrenVisiting,Designation,MonthlyIncome"
]


tokorean = ['샘플 아이디', '나이','고객의 제품 인지 방법 ', '주거 중인 도시의 등급', '영업 사원이 고객에게 제공하는 프레젠테이션 기간',
                '직업', '성별', '고객과 함께 여행을 계획 중인 총 인원', '영업 사원의 프레젠테이션 후 이루어진 후속 조치 수',
                '영업 사원이 제시한 상품', '선호 호텔 숙박업소 등급', '결혼여부', '평균 연간 여행 횟수', '여권 보유 여부 (0: 없음, 1: 있음)',
                '영업 사원의 프레젠테이션 만족도', '자동차 보유 여부', '함께 여행을 계획 중인 5세 미만의 어린이 수', '(직업의) 직급',
                '월 급여',]