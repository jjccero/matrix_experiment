from math import sin
from typing import Callable

import numpy as np


# 由于fx是奇函数，对-2到2积分为0

def I(f: Callable, a, b, n: int):
    h = (b - a) / n
    I_ = np.zeros(1, np.float32)
    I_[0] = (f(a) + f(b)) / 2
    for i in range(1, n):
        I_[0] += f(a + i * h)
    I_[0] *= h
    return I_[0]


f = lambda x: x * x * sin(x)
print(I(f, a=-2, b=2, n=20))
print(I(f, a=-2, b=2, n=40))
print(I(f, a=-2, b=2, n=80))
print(I(f, a=-2, b=2, n=200))
