import numpy as np
import matplotlib.pyplot as plt

def selu():
    return np.maximum(0,x) + np.minimum(0,1.0507*(np.exp(x)-1))

selu2 = lambda x: np.maximum(0,x) + np.minimum(0,1.0507*(np.exp(x)-1))

x = np.arange(-7, 5, 0.10)
y = selu2(x)

plt.plot(x,y,'k-')
plt.grid()
plt.show()