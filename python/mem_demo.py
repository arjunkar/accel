"""
Two theoretically equivalent matrix multiplication procedures
are given: serial1 and serial2.

However, their runtimes show a large difference (compare.pdf).
This difference is due to the internal operation of the computer system.
It is easier to explore such ideas in a lower level language
like C or C++.
"""

import numpy as np
from timing import time_square
from matplotlib import pyplot as plt

def serial1(A, B):
    rows = A.shape[0]
    cols = B.shape[1]
    hid = A.shape[1]
    C = np.zeros(shape=(rows,cols))
    for i in range(rows):
        for j in range(cols):
            for k in range(hid):
                C[i,j] += A[i,k] * B[k,j]
    return C

def serial2(A, B):
    rows = A.shape[0]
    cols = B.shape[1]
    hid = A.shape[1]
    C = np.zeros(shape=(rows,cols))
    for i in range(rows):
        for j in range(cols):
            out = 0
            for k in range(hid):
                out += A[i,k] * B[k,j]
            C[i,j] = out
    return C

# Testing

# x_data = range(10, 300+1, 10)
# ser1_data = [time_square(x, serial1) for x in x_data]
# ser2_data = [time_square(x, serial2) for x in x_data]

# print(x_data)
# print("\n")
# print(ser1_data)
# print("\n")
# print(ser2_data)