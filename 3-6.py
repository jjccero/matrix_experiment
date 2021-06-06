import numpy as np
import matplotlib.pylab as plt
from typing import Callable


def newton(f: Callable, df: Callable, x, t=20):
    x_ = x
    es = []

    for i in range(t):
        x = x - f(x) / df(x)
        e = abs(x - x_)
        es.append(e)
        x_ = x
        if e == 0.0:
            break
    r2 = []
    r1 = []
    for i in range(1, len(es)):
        r2.append(es[i] / es[i - 1] / es[i - 1])
        r1.append(es[i] / es[i - 1])
    return x, r2, r1


def secant(f: Callable, x0, x1, t=20):
    es = [abs(x1 - x0)]
    for i in range(t):
        f1 = f(x1)
        f0 = f(x0)
        if es[-1] == 0.0:
            break
        x2 = x1 - (x1 - x0) * f1 / (f1 - f(x0))
        x0 = x1
        es.append(abs(x2 - x1))
        x1 = x2

    r1 = []
    for i in range(1, len(es)):
        r1.append(es[i] / es[i - 1])
    return x1, r1


f = lambda x: (((((54 * x + 45) * x - 102) * x - 69) * x + 35) * x + 16) * x - 4
df = lambda x: ((((54 * 6 * x + 45 * 5) * x - 102 * 4) * x - 69 * 3) * x + 35 * 2) * x + 16

x_ = np.linspace(-2., 2., 401)


print('newton')
plt.plot(x_, f(x_))
plt.title('newton')
for j, x in enumerate([-2., -1., 0.25, 0.6, 2.]):
    x, r2, r1 = newton(f, df, x)
    print('x{} = {}, e(k+1)/e(k)^2 = {}'.format(j + 1, x, r2))
    print('x{} = {}, e(k+1)/e(k) = {}'.format(j + 1, x, r1))
    plt.scatter(x, f(x), label='x{} = {}'.format(j + 1, x))
plt.legend()
plt.show()

print('secant')
plt.title('secant')
plt.plot(x_, f(x_))
for j, x in enumerate([-2., -1., 0.25, 0.6, 2.]):
    x, r1 = secant(f, x + 0.1, x)
    print('x{} = {}, e(k+1)/e(k) = {}'.format(j + 1, x, r1))
    plt.scatter(x, f(x), label='x{} = {}'.format(j + 1, x))
plt.legend()
plt.show()
