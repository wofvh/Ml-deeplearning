import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x) # tanh함수는 -1~1사이의 값으로 변환

plt.plot(x,y,'k-')
plt.grid()
plt.show()
