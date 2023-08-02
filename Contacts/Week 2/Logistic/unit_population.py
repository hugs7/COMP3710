# Unit test for computing one

import numpy as np
import matplotlib.pyplot as plt

N = 30
x = 0.5
l = 2.3


T = np.arange(N)
X = np.zeros(N)

for t in T:
    x = l*x*(1-x)
    X[t] = x

plt.plot(T, X, '-b')

plt.show()