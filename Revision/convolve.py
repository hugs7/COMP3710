import numpy as np
import scipy.signal as sps
import timeit

l1 = np.random.rand(100000)
l2 = np.random.rand(100)


def perform_convolve(conv_func, l1, l2):
    def wrapper():
        return conv_func(l1, l2)

    exec_time = timeit.timeit(wrapper, number=10000)  # Adjust the number as needed
    print("Time taken: {:.6f} seconds".format(exec_time))

    return exec_time


funcs = [
    # (sps.fftconvolve, "Signal FFT convolve"),
    (sps.convolve, "Signal convolve"),
    (np.convolve, "numpy convolve"),
]

for func, name in funcs:
    print(name)
    print(perform_convolve(func, l1, l2))
    print("----")
