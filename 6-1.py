import math
from typing import Callable

import matplotlib.pylab as plt
import numpy as np


def euler(f: Callable, a, b, u0, h):
    n = int((b - a) / h) + 1
    u = np.zeros(n)
    x = np.linspace(a, b, n)
    u[0] = u0
    for t in range(n - 1):
        u[t + 1] = u[t] + h * f(x[t], u[t])
    return x, u


def euler_improved(f: Callable, a, b, u0, h):
    n = int((b - a) / h) + 1
    u = np.zeros(n)
    x = np.linspace(a, b, n)
    u[0] = u0
    for t in range(n - 1):
        ft = f(x[t], u[t])
        u[t + 1] = u[t] + h / 2 * (ft + f(x[t + 1], u[t] + h * ft))
    return x, u


def runge_kutta(f: Callable, a, b, h, u0):
    n = int((b - a) / h) + 1
    u = np.zeros(n)
    x = np.linspace(a, b, n)
    u[0] = u0
    for t in range(n - 1):
        k1 = f(x[t], u[t])
        k2 = f(x[t] + h / 2, u[t] + h * k1 / 2)
        k3 = f(x[t] + h / 2, u[t] + h * k2 / 2)
        k4 = f(x[t] + h, u[t] + h * k3)
        u[t + 1] = u[t] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, u


f = lambda x, u: (2 * u / x + x * x * math.exp(x))

for h in [0.1, 0.05, 0.01]:
    plt.figure()
    plt.title('h = {}'.format(h))
    x, u = euler(f, a=1, b=2, u0=0, h=h)
    plt.plot(x, u)
    x, u = euler_improved(f, a=1, b=2, u0=0, h=h)
    plt.plot(x, u)
    x, u = runge_kutta(f, a=1, b=2, u0=0, h=h)
    plt.plot(x, u)
    plt.legend(["Euler", "improved Euler", "Runge-Kutta"])
    plt.show()
