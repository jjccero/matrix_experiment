import numpy as np
import matplotlib.pylab as plt


def spline(x: np.ndarray, xs: np.ndarray, ys: np.ndarray, m0, mn):
    dim = xs.shape[0]
    h = xs[1:] - xs[:-1]
    lam = h[1:] / (h[1:] + h[:-1])
    miu = h[:-1] / (h[1:] + h[:-1])
    g = 3 * (miu * (ys[2:] - ys[1:-1]) / h[1:] + lam * (ys[1:-1] - ys[:-2]) / h[:-1])
    m = np.zeros_like(xs)
    m[0] = m0
    m[-1] = mn
    b = g.copy()
    b[0] -= lam[0] * m0
    b[-1] -= miu[-1] * mn
    b.reshape((dim - 2, 1))
    A = np.zeros((dim - 2, dim - 2))
    A[0, 0] = 2
    for k in range(1, dim - 2):
        A[k, k] = 2
        A[k - 1, k] = miu[k - 1]
        A[k, k - 1] = lam[k - 1]
    mx = np.linalg.solve(A, b).reshape(dim - 2)
    m[1:-1] = mx
    y = np.zeros_like(x)

    for i, x_ in enumerate(x):
        for k in range(dim - 1):
            if xs[k] <= x_ <= xs[k + 1]:
                s, t = x_ - xs[k], x_ - xs[k + 1]
                hk = h[k]
                y[i] = ((hk + 2 * s) * t * t * ys[k] + (hk - 2 * t) * s * s * ys[k + 1]) / hk ** 3 + (
                        s * t * t * m[k] + t * s * s * m[k + 1]) / hk ** 2
                break
    return y


f = lambda x: 1 / (1 + x ** 2)
df = lambda x: -2 * x / ((1 + x ** 2) ** 2)

a = -5
b = 5
x = np.linspace(a, b, 101, dtype=np.float32)
# 第一类边界条件
m0 = df(a)
mn = df(b)
plt.plot(x, f(x), label='y=1/(1+x^2)')
for n in [5, 10, 20]:
    xs = np.linspace(a, b, n + 1, dtype=np.float32)
    ys = f(xs)
    y = spline(x, xs, ys, m0, mn)
    plt.plot(x, y, label='interval = {}'.format(10 / n))

plt.legend()
plt.show()
