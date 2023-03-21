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
demonstrates how a multithreaded Python program can lead to
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
 81           0 SETUP_LOOP              52 (to 54)
              2 LOAD_GLOBAL              0 (range)
              4 LOAD_CONST               1 (1000000)
              6 CALL_FUNCTION            1
              8 GET_ITER
        >>   10 FOR_ITER                40 (to 52)
             12 STORE_FAST               1 (_)

 82          14 LOAD_GLOBAL              1 (lock)
             16 SETUP_WITH              26 (to 44)
             18 POP_TOP

 86          20 LOAD_GLOBAL              2 (shared_balance)
             22 STORE_FAST               2 (balance)

 87          24 LOAD_FAST                2 (balance)
             26 LOAD_FAST                0 (sgn)
             28 LOAD_CONST               2 (100)
             30 BINARY_MULTIPLY
             32 INPLACE_ADD
             34 STORE_FAST               2 (balance)

 88          36 LOAD_FAST                2 (balance)
             38 STORE_GLOBAL             2 (shared_balance)
             40 POP_BLOCK
             42 LOAD_CONST               0 (None)
        >>   44 WITH_CLEANUP_START
             46 WITH_CLEANUP_FINISH
             48 END_FINALLY
             50 JUMP_ABSOLUTE           10
        >>   52 POP_BLOCK
        >>   54 LOAD_CONST               0 (None)
             56 RETURN_VALUE
None
"""

import threading
import dis

# Change this boolean to try the two calculations
locked = False

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
    for _ in range(1000000):
        with lock:
        # If locked == False, this context does nothing
        # and the following ops may be interrupted
            global shared_balance
            balance = shared_balance
            balance += sgn*100
            shared_balance = balance

# Uncomment to see disassembled bytecode of step function.
# print(dis.dis(step))

threads = [
        threading.Thread(target=step, args=(1,)),
        threading.Thread(target=step, args=(-1,))
        ]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(shared_balance)