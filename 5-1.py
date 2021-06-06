import math
from typing import Callable

import numpy as np


def I_trapezoid(f: Callable, a, b, n: int):
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


a = 0
b = math.pi
f = lambda x: x * x * math.sin(x)
I = math.pi ** 2 - 4
print('复化梯形')
for n in [20, 40, 80, 200]:
    I_ = I_trapezoid(f, a, b, n=n)
    print('{}\t{}\t{}'.format(n, I_, abs(I - I_)))
print('复化Simpson')
for n in [20, 40, 80, 200]:
    I_ = I_simpson(f, a, b, n=n)
    print('{}\t{}\t{}'.format(n, I_, abs(I - I_)))
