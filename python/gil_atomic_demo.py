"""
A mild obstruction to thread-based parallelism in CPython is
the Global Interpreter Lock (GIL).

This is a mutex which allows only a single thread at a time
to execute Python bytecode.
However, at any point in the execution of a standard Python
program, the thread may give up the GIL or may be forced to
do so.
This means that the GIL does not ensure thread-safety of
Python programs for free, and we must use locking primitives
just like we need to when we spawn e.g. POSIX threads in C.

This program, taken from the helpful answer here:
https://stackoverflow.com/questions/105095/are-locks-unnecessary-in-multi-threaded-python-code-because-of-the-gil?rq=1
demonstrates how a mutlithreaded Python program can lead to
incorrect results if not locked properly.

The result of the calculation of shared_balance is expected
to be zero since there are 1e6 increments of 100 and 1e6
decrements of 100.
With a lock, this is guaranteed.
Without a lock, the GIL may be relinquished by a thread in
the middle of an increment or decrement (which, as the dis module
shows, involves several Python bytecode operations).
This leads to overwriting and missed updates, and the result is
generically nonzero.
Only individual Python bytecode ops should be considered atomic.

Change the locked boolean to test the two versions.
locked = True gives correct results while
locked = False gives incorrect results.

For reference, the dis.dis(step) output is:
56           0 LOAD_GLOBAL              0 (lock)
              2 SETUP_WITH              26 (to 30)
              4 POP_TOP

 60           6 LOAD_GLOBAL              1 (shared_balance)
              8 STORE_FAST               1 (balance)

 61          10 LOAD_FAST                1 (balance)
             12 LOAD_FAST                0 (sgn)
             14 LOAD_CONST               1 (100)
             16 BINARY_MULTIPLY
             18 INPLACE_ADD
             20 STORE_FAST               1 (balance)

 62          22 LOAD_FAST                1 (balance)
             24 STORE_GLOBAL             1 (shared_balance)
             26 POP_BLOCK
             28 LOAD_CONST               0 (None)
        >>   30 WITH_CLEANUP_START
             32 WITH_CLEANUP_FINISH
             34 END_FINALLY
             36 LOAD_CONST               0 (None)
             38 RETURN_VALUE
None
"""

import threading
import dis

# Change this boolean to try the two calculations
locked = True

# Trivial "fake" context manager for unlocked version
class TrivialLock():
    def __enter__(self):
        pass
    def __exit__(*args):
        pass

# Real lock to be used as context manager if locked
lock = threading.Lock() if locked else TrivialLock()

# Global variable which should be zero at the end
shared_balance = 0

def step(sgn):
    with lock: 
    # If locked == False, this context does nothing
    # and the following ops may be interrupted
        global shared_balance
        balance = shared_balance
        balance += sgn*100
        shared_balance = balance

# Uncomment to see disassembled bytecode of step function.
# print(dis.dis(step))

class Deposit(threading.Thread):
    def run(self):
        for _ in range(1000000):
            step(1)

class Withdraw(threading.Thread):
    def run(self):
        for _ in range(1000000):
            step(-1)

threads = [Deposit(), Withdraw()]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(shared_balance)