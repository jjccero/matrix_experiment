import numpy as np


def exact_result(N: int):
    return 1 / 2 * (3 / 2 - 1 / N - 1 / (N + 1))


def large2small(N: int):
    s = np.zeros(1, np.float32)
    j = 2
    while j <= N:
        s += 1 / (j * j - 1)
        j += 1
    return s[0]


def small2large(N: int):
    s = np.zeros(1, np.float32)
    j = N
    while j >= 2:
        s += 1 / (j * j - 1)
        j -= 1
    return s[0]


for N in [10 ** 2, 10 ** 4, 10 ** 6]:
    exact_ = exact_result(N)
    l_ = large2small(N)
    s_ = small2large(N)
    print('S{},从大到小,结果{},误差{:e}'.format(N, l_, exact_ - l_))
    print('S{},从小到大,结果{},误差{:e}'.format(N, s_, exact_ - s_))

# 通过本上机题，我明白了多个数相加时要做到更多的有效位数，应把绝对值小的数先相加，再与绝对值大的数相加
