from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder

text1 = '진짜 매우 나는 나는 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'
text2 = '나는 지구용사 이재근이다. 멌있다. 또 얘기해봐 '

token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)


x = token.texts_to_sequences([text1,text2])
print(x)
# [[4, 2, 3, 3, 2, 5, 6, 7, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

x_new = x[0] + x[1]
print(x_new)


x_new = to_categorical(x_new)
print(x_new)
print(x_new.shape)  

# [[4, 3, 1, 1, 3, 5, 6, 7, 2, 2, 2, 8], [1, 9, 10, 11, 12, 13]]
# [4, 3, 1, 1, 3, 5, 6, 7, 2, 2, 2, 8, 1, 9, 10, 11, 12, 13]
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# (18, 14)

ohe = OneHotEncoder(categories= 'auto',sparse=True)
x = ohe.fit_transform(x.reshape(-1,1,0))
x = np.array(x_new)
x = x.reshape(-1, 1)
onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
