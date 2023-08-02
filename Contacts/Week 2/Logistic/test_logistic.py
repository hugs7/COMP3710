# Unit test for computing one

import numpy as np
import matplotlib.pyplot as plt

def logistic(l, x):
    return l*x*(1-x)

N = 5
x0 = 0.5
last = 100
Lambdas = np.linspace(0.5, 3.7, 1000)
x = x0 * np.ones_like(Lambdas)
T = np.arange(N)
X = np.zeros(N)

for t in T:
    x = logistic(Lambdas, x)
    if t >= (N-last):
        plt.plot(Lambdas, x, ',k', alpha=.75)
    

plt.show()