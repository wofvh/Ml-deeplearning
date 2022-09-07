import numpy as np
import matplotlib.pyplot as plt

def softmax(x):  #softmax함수는 0~1사이의 값으로 변환 # 0보다 작으면 0, 0보다 크면 x # 0이하는 0으로 만들어줌
    return np.exp(x) / np.sum(np.exp(x))

softmax2 = lambda x: np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()