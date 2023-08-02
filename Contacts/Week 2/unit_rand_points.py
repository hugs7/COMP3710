# Unit test for random points

import numpy as np
import matplotlib.pyplot as plt

n = 3
r = np.arange(0,n)
res = 300

unit_circle = np.exp((2.0*np.pi*r*1j)/res)   # Define unit circle

# Pick starting position
start = np.random.randint(0,n)

plt.plot(np.real(unit_circle), np.imag(unit_circle), "b-")
plt.plot(np.real(points), np.imag(points), "r-")


plt.show()