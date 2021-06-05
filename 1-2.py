from typing import List

import numpy as np


# cs为系数向量，次数从大到小排列
def f(cs: List, x):
    s = np.zeros(1, np.float32)
    s[0] = 0
    for c in cs:
        s = x * s + c
    return s[0]


print(f(cs=[7, 3, -5, 11], x=23))
