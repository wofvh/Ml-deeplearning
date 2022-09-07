import numpy as np
import matplotlib.pyplot as plt

def relu(x):  # relu함수는 0~1사이의 값으로 변환 # 0보다 작으면 0, 0보다 크면 x # 0이하는 0으로 만들어줌
    return np.maximum(0,x)

relu2 = lambda x: np.maximum(0,x) #maximum은 두개의 값을 비교해서 큰 값을 반환해줌

x = np.arange(-5, 5, 0.1)
y = relu2(x)

plt.plot(x,y,'k-')
plt.grid()
plt.show()

#elu, selu, leakyrelu, prelu, rrelu, thresholdedrelu 