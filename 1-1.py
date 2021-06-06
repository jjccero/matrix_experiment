import numpy as np

f = lambda x: 1 / 2 * (3 / 2 - 1 / x - 1 / (x + 1))


def fl2s(n: int):
    s = np.zeros(1, np.float32)
    for j in range(2, n + 1):
        s += 1 / (j * j - 1)
    return s[0]


def fs2l(n: int):
    s = np.zeros(1, np.float32)
    for j in reversed(range(2, n + 1)):
        s += 1 / (j * j - 1)
    return s[0]


for n in [10 ** 2, 10 ** 4, 10 ** 6]:
    f0 = f(n)
    f1 = fl2s(n)
    f2 = fs2l(n)
    print('S{} = {}, 从大到小,{},{},从小到大,{},{}'.format(n, f0, f1, abs(f1 - f0), f2, abs(f2 - f0)))
