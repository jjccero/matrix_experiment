from typing import Callable

import matplotlib.pylab as plt
import numpy as np


def linear_fit(f: Callable, x: np.ndarray):
    dim = x.shape[0]
    x = np.linspace(0, 1, dim)
    y = f(x)
    A = np.array([dim, x.sum(), x.sum(), (x * x).sum()], dtype=x.dtype).reshape((2, 2))
    b = np.array([y.sum(), (y * x).sum()]).reshape((2, 1))
    x = np.linalg.solve(A, b)
    return x[1, 0], x[0, 0]


f = lambda x: x ** 2
x = np.linspace(0, 1, 101)
plt.plot(x, f(x), label='y=x^2')

for n in [5, 10, 15, 25, 30]:
    x = np.linspace(0, 1, n + 1, dtype=np.float)
    a, b = linear_fit(f, x)
    f1 = lambda x: a * x + b
    y = f1(x)
    plt.plot(x, y, label='{}, y={:.5f}x{:.5f}'.format(n, a, b))

plt.legend()
plt.show()
