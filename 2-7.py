import numpy as np


def jocobi(A: np.ndarray, b: np.ndarray, eps=1e-6):
    dim = A.shape[0]
    x_ = np.zeros((dim, 1))
    x = np.zeros((dim, 1))
    t = 0
    while True:
        t += 1
        for i in range(dim):
            xs = 0.0
            for j in range(dim):
                if i != j:
                    xs += A[i, j] * x_[j, 0]
            x[i, 0] = (b[i, 0] - xs) / A[i, i]
        if np.linalg.norm(x - x_) < eps:
            break
        x_[:, :] = x[:, :]
    return x.reshape(dim), t


def gauss_seidel(A: np.ndarray, b: np.ndarray, eps=1e-6):
    dim = A.shape[0]
    x_ = np.zeros((dim, 1))
    x = np.zeros((dim, 1))
    t = 0
    while True:
        t += 1
        for i in range(dim):
            xs = 0.0
            for j in range(dim):
                if i != j:
                    xs += A[i, j] * x[j, 0]
            x[i, 0] = (b[i, 0] - xs) / A[i, i]
        if np.linalg.norm(x - x_) < eps:
            break
        x_[:, :] = x[:, :]
    return x.reshape(dim), t


for n in [10, 20, 30, 50, 100]:
    A = np.zeros((n, n))
    b = np.ones((n, 1))
    b[0, 0] = 2
    b[-1, -1] = 2
    for i in range(n):
        A[i, i] = 3
        if i < n - 1:
            A[i, i + 1] = -1
            A[i + 1, i] = -1
    x1, t1 = jocobi(A, b)
    print('n={},Jocobi,{}次,x={}'.format(n, t1, x1))
    x2, t2 = gauss_seidel(A, b)
    print('n={},Gauss-Seidel,{}次,x={}'.format(n, t2, x2))
