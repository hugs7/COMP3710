# Unit test for random points

import numpy as np
import matplotlib.pyplot as plt

n = 3
r = np.arange(0,n)
res = 300

points = np.exp((2.0*np.pi*r*1j)/res)
print(points)
exit()

start = 0.1+0.5j

def compute_new_rand_location(point):
    rand_location = np.random.randint(0,n)
    vector = (points[rand_location])
    print(vector)

    # Pick starting position
    next_point = point + vector

    return next_point, rand_location

next_point = start
iterations = 1000
for iteration in range(iterations):
    next_point, rand_location = compute_new_rand_location(next_point)
    plt.plot(np.real(next_point), np.imag(next_point), "b.")

plt.plot(np.real(points), np.imag(points), "r.")
plt.plot(np.real(start), np.imag(start), "g.")

plt.show()





def compute_new_rand_location(startLoc):
    rand_location = np.random.randint(0,n)
