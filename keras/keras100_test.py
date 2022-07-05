from pydoc import describe
import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook

# path = './_data/kaggle_titanic/'

# train = pd.read_csv(path +'train.csv',index_col=0)
# test = pd.read_csv(path + 'test.csv',index_col=0)

# print(train)#[891 rows x 11 columns]
# print(test) #[418 rows x 10 columns]

path = './_data/kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
print(train_set) # [891 rows x 11 columns]
print(train_set.describe())
print(train_set.info())


print(test_set) # [418 rows x 10 columns]
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
print(test_set.isnull().sum())

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 0, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print(train_set) 
print(train_set.isnull().sum())

# test_set.drop(drop_cols, axis = 1, inplace =True)
# cols = ['Name','Sex','Ticket','Embarked']
# for col in tqdm_notebook(cols):
#     le = LabelEncoder()
#     train_set[col]=le.fit_transform(train_set[col])
#     test_set[col]=le.fit_transform(test_set[col])
# x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
# print(x) #(891, 9)
# y = train_set['Survived']
# print(y.shape) #(891,)


'''
x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=31
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(9, activation='swish', input_dim=12))
model.add(Dense(17, activation='elu'))
model.add(Dense(19, activation='swish'))
model.add(Dense(14, activation='elu'))
model.add(Dense(1))

#3. 컴파일, 훈련

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1, 
                              restore_best_weights=True)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=80, batch_size=100, verbose=1,validation_split=0.2, callbacks=[earlyStopping])

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
'''