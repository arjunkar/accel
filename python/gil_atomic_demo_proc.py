
"""
A similar demonstration of the GIL being exchanged between
processes.

This takes quite a bit longer than the thread demonstration. 
Perhaps the reason is that the processor must perform 
process context switches each time the GIL is exchanged, 
which are more costly than thread context swaps.
"""

import multiprocessing as mp

# Change this boolean to try the two calculations
locked = True

# Trivial "fake" context manager for unlocked version
class TrivialLock():
    def __enter__(self):
        pass
    def __exit__(*args):
        pass

# Real lock to be used as context manager if locked
lock = mp.Lock() if locked else TrivialLock()

# Global variable which should be zero at the end.
# To share between processes, we need mp.Value.
shared_balance = mp.Value('i', 0, lock=False)
# Typecode 'i' is signed integer.
# lock=False makes this shared value process-unsafe.
# Setting it to True would essentially eliminate the need
# for our own lock.

def step(shared_balance, sgn):
    for _ in range(1000000):
        with lock:
        # If locked == False, this context does nothing
        # and the following ops may be interrupted
            balance = shared_balance.value
            balance += sgn*100
            shared_balance.value = balance

processes = [
        mp.Process(target=step, args=(shared_balance, 1,)),
        mp.Process(target=step, args=(shared_balance, -1,))
        ]

for process in processes:
    process.start()

for process in processes:
    process.join()

print(shared_balance)