"""
Testing tiling method in Python.
"""

import numpy as np

def aware(A, B, row, col, hid, BLOCK_SIZE):
    
    row_block = row // BLOCK_SIZE
    col_block = col // BLOCK_SIZE
    hid_block = hid // BLOCK_SIZE
    A = A.flatten()
    B = B.flatten()
    C = np.zeros(shape=(row,col)).flatten()
    
    for ib in range(row_block):
        for jb in range(col_block):
            for kb in range(hid_block):
                start_C = ib * BLOCK_SIZE * col + jb * BLOCK_SIZE
                start_A = ib * BLOCK_SIZE * hid + kb * BLOCK_SIZE
                start_B = kb * BLOCK_SIZE * col + jb * BLOCK_SIZE
                for i in range(BLOCK_SIZE):
                    for j in range(BLOCK_SIZE):
                        for k in range(BLOCK_SIZE):
                            C[start_C + i * col + j] += A[start_A + i * hid + k] * B[start_B + k * col + j]

    return C.reshape((row,col))