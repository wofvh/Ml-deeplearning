from unittest import result
import numpy as np
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import true
from tensorboard import summary
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

#1.데이터
datasets = load_boston()
