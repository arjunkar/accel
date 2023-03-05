"""
Timing function for testing matrix multiplication.

Expects f(A, B) to create and return the product C = A @ B.
"""

import numpy as np
import time

def time_square(dim, fun):
    # Initialize matrices
    A = np.random.randint(low=1, high=5, size=(dim,dim))
    B = np.random.randint(low=1, high=5, size=(dim,dim))

    # Timed multiplication
    s = time.time()
    C = fun(A, B)
    f = time.time()

    # Assert output correctness
    assert((C == A@B).all())

    # Report time
    # print("--- %s seconds ---" %(f-s))
    return f-s