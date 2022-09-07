import numpy as np
import matplotlib.pyplot as plt

def elu(x):  # elu함수는 0~1사이의 값으로 변환 # 0보다 작으면 0, 0보다 크면 x # 0이하는 0으로 만들어줌
    return np.maximum(0,x) + np.minimum(0,0.1*(np.exp(x)-1))

elu2 = lambda x: np.maximum(0,x) + np.minimum(0,0.1*(np.exp(x)-1))

x = np.arange(-7, 5, 0.10)
y = elu2(x)

plt.plot(x,y,'k-')
plt.grid()
plt.show()