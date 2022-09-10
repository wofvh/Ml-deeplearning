import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):  
    return 1 / (1 + np.exp(-x))

sigmoi2 = lambda x: 1 / (1 + np.exp(-x)) #람다식 0~1사이의 값으로 변환

x = np.arange(-6, 5, 0.8)
print(x)
print(len(x)) #100

y = sigmoid(x) #시그모이드 함수에 x를 넣어서 y를 구함

plt.plot(x,y,'k-')
plt.grid()
plt.show()