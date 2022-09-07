import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoi2 = lambda x: 1 / (1 + np.exp(-x))