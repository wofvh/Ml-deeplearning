from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

#1.데이터
datasets = load_boston()
x,y = datasets.data, datasets.target
print(x.shape,y.shape)  #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1234, train_size=0.8,)


#2.모델구성

model = make_pipeline(StandardScaler(),
                      LinearRegression())


model.fit(x_train,y_train)

print(model.score(x_test,y_test))


#결과 0.7665382927362877


