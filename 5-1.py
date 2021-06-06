from math import sin
from typing import Callable

import numpy as np


# 由于fx是奇函数，对-2到2积分为0

def I(f: Callable, a, b, n: int):
    h = (b - a) / n
    I_ = np.zeros(1, np.float32)
    I_[0] = f(a) + f(b)
    for i in range(1, n):
        I_[0] += 2 * f(a + i * h)
    I_[0] *= h / 2
    return I_[0]


def I_simpson(f: Callable, a, b, n: int):
    n = n // 2
    h = (b - a) / n
    I_ = np.zeros(1, np.float32)
    I_[0] = f(a) + f(b)
    for i in range(n):
        I_[0] += 4 * f(a + (i + 0.5) * h)
    for i in range(1, n):
        I_[0] += 2 * f(a + i * h)
    I_[0] *= h / 6
    return I_[0]


f = lambda x: x * x * sin(x)
print(I_simpson(f, a=-2, b=2, n=20))
print(I(f, a=-2, b=2, n=40))
print(I(f, a=-2, b=2, n=80))
print(I(f, a=-2, b=2, n=200))
