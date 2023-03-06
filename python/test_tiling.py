import numpy as np
from tiling import aware

def test_tiling():
    def test_dims(d_in, d_hid, d_out, BLOCK):
        A = np.random.randint(low=1, high=5, size=(d_in,d_hid))
        B = np.random.randint(low=1, high=5, size=(d_hid,d_out))

        C = aware(A, B, d_in, d_out, d_hid, BLOCK)
        assert((C == A@B).all())

    test_dims(30,30,30,5)
    test_dims(10,20,30,5)
    test_dims(30,10,15,5)
    test_dims(100,80,60,20)
    test_dims(5*17, 5*6, 5*12, 5)