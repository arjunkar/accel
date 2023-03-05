"""
Serial and parallel matrix multiplication with multiprocessing.

As multiprocessing.cpu_count() reveals, there are 4 cores
available for Python processes on the Intel chip:
(Intel(R) Core(TM) i5-5257U CPU @ 2.70GHz).

In fact, the i5-5257U is listed as having only 2 physical
cores.  The other 2 are likely due to Intel's hyperthreading.

Analysis:
dim (N) = 700 is where the parallel method begins to outpace
the serial method despite the memory inefficiency.
The advantage of 4 simultaneous execution contexts will
eventually dominate over a linear memory slowdown.

The reason for this behavior is the scaling of the asymptotic work.
The serial work, assuming no memory inefficiency, is:
Work_s = (Work_dot) * (N**2) = 2*N**3
The parallel work, assuming only the linear memory slowdown, is:
Work_p = (Work_dot + Work_memcopy) * (N**2) = 4*N**3
The serial procedure takes a time equal to its work, but
the parallel procedure takes time Work_p / num_proc = N**3.
Of course, we do not see an immediate 2x improvement when using
the parallel method on small inputs because of the overhead costs
associated with multiprocessing.
But by N = 800, time_s = 186.9 sec and time_p = 175.1 sec.
"""

import numpy as np
from timing import time_square
import multiprocessing as mp


def dot(v1, v2):
    out = 0
    for i in range(v1.shape[0]):
        out += v1[i] * v2[i]
    return out

def serial(A, B):
    # requires dimensions allowing C = A @ B
    rows = A.shape[0]
    cols = B.shape[1]
    C = [[dot(A[i], B[:,j]) for j in range(cols)] for i in range(rows)]
    # Because all calls to dot share the same memory space, the serial
    # implementation is very access-efficient.
    return np.array(C)

def parallel(A, B):
    num_workers = mp.cpu_count() # 4 available cores on an i5-5257U
    rows = A.shape[0]
    cols = B.shape[1]
    with mp.Pool(processes=num_workers) as pool:
        C = [[pool.apply_async(dot, args=(A[i], B[:,j])) for j in range(cols)]
                for i in range(rows)]
        # Each call to apply_async copies the arguments into the
        # subprocess, leading to a slowdown compared to the serial
        # evaluation which shares memory.
        return np.array(
            [[C[i][j].get() for j in range(cols)] for i in range(rows)]
        )

# Testing
# time_square(800, serial)
# time_square(800, parallel)