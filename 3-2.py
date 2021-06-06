import numpy as np
from typing import Callable


def binsolve(f: Callable, a, b, t):
    x = np.zeros(t, dtype=np.float32)
    for i in range(t):
        m = (a + b) / 2
        x[i] = m
        fa, fm, fb = f(a), f(m), f(b)
        if fa * fm <= 0:
            b = m
        elif fb * fm <= 0:
            a = m
    return x


f = lambda x: x * np.cos(x) + 2

t = 50
x_ = binsolve(f, a=-4, b=4, t=t)
print(x_)
