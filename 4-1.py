import numpy as np
import matplotlib.pylab as plt


def lagrange(x: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    dim = xs.shape[0]
    y = np.zeros_like(x)
    for k in range(dim):
        xi = xs[k]
        li = np.ones_like(x)
        for j in range(dim):
            if k != j:
                xj = xs[j]
                li *= (x - xj) / (xi - xj)
        y += li * ys[k]
    return y


f = lambda x: 1 / (1 + x ** 2)

x = np.linspace(-5, 5, 101, dtype=np.float32)

for n in [5, 10, 20]:
    xs = np.linspace(-5, 5, n+1, dtype=np.float32)
    ys = f(xs)
    y = lagrange(x, xs, ys)
    plt.plot(x, y, label='interval = {}'.format(10/n))
    plt.plot(x, f(x), label='y=1/(1+x^2)')
    plt.legend()
    plt.show()
