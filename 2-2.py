import numpy as np


def cholesky(A: np.ndarray):
    dim = A.shape[0]

    L = np.zeros_like(A)
    for j in range(dim):
        ljj = A[j, j]
        for k in range(j):
            ljj -= L[j, k] ** 2
        ljj **= 0.5
        L[j, j] = ljj
        for i in range(j + 1, dim):
            lij = A[i, j]
            for k in range(j):
                lij -= L[i, k] * L[j, k]
            lij /= ljj
            L[i, j] = lij
    return L


def solve(L: np.ndarray, U: np.ndarray, b: np.ndarray):
    # Ly=b
    y = np.zeros_like(b)
    dim = b.shape[0]
    for i in range(dim):
        rows = b[i, 0]
        for j in range(i):
            rows -= y[j, 0] * L[i, j]
        y[i, 0] = rows / L[i, i]
    # Ux=y
    x = np.zeros_like(b)
    for i in reversed(range(dim)):
        rows = y[i, 0]
        for j in range(i + 1, dim):
            rows -= x[j, 0] * U[i, j]
        x[i, 0] = rows / U[i, i]
    return x


A = np.asarray(
    [
        7, 1, -5, 1,
        1, 9, 2, 7,
        -5, 2, 7, -1,
        1, 7, -1, 9
    ],
    dtype=np.float32
).reshape((4, 4))
b = np.array([13, -9, 6, 0], dtype=np.float32).reshape((4, 1))

L = cholesky(A)
x = solve(L, L.T, b)

print('L:\n{}'.format(L))
print('x:\n{}'.format(x))

