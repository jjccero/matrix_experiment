import numpy as np


def LU(A: np.ndarray):
    dim = A.shape[0]

    L = np.eye(dim, dtype=A.dtype)
    U = A.copy()

    for i in range(dim):
        uii = U[i, i]
        for j in range(i + 1, dim):
            lji = U[j, i] / uii
            L[j, i] = lji
            for k in range(i, dim):
                U[j, k] -= lji * U[i, k]
    return L, U


def PLU(A: np.ndarray):
    dim = A.shape[0]

    maxis = []
    Ls = []
    U = A.copy()

    for i in range(dim - 1):
        maxi = np.argmax(np.abs(U[i:, i])) + i
        maxis.append(maxi)
        U[[i, maxi], :] = U[[maxi, i], :]
        L = np.eye(dim, dtype=A.dtype)
        Ls.append(L)
        uii = U[i, i]
        for j in range(i + 1, dim):
            lji = U[j, i] / uii
            L[j, i] = lji
            for k in range(i, dim):
                U[j, k] -= lji * U[i, k]

    P = np.eye(dim, dtype=A.dtype)
    for i in range(dim - 1):
        maxi = maxis[i]
        L = Ls[i]
        for j in range(i + 1, dim - 1):
            maxj = maxis[j]
            L[[j, maxj], :] = L[[maxj, j], :]
            L[:, [j, maxj]] = L[:, [maxj, j]]

        P[[i, maxi], :] = P[[maxi, i], :]
    for i in range(1, dim - 1):
        Ls[i] = np.matmul(Ls[i - 1], Ls[i])
    return P, Ls[-1], U


def det(A: np.ndarray):
    L, U = LU(A)
    dim = U.shape[0]

    det_ = 1.0
    for i in range(dim):
        det_ *= U[i, i]
    return det_


def solve(L: np.ndarray, U: np.ndarray, b: np.ndarray):
    # Ly=b
    y = np.zeros_like(b)
    dim = b.shape[0]
    for i in range(dim):
        rows = b[i, 0]
        for j in range(0, i):
            rows -= b[j, 0] * L[i, j]
        y[i] = rows / L[i, i]
    # Ux=y
    x = np.zeros_like(b)
    for i in reversed(range(dim)):
        rows = y[i]
        for j in range(i + 1, dim):
            rows -= x[j] * U[i, j]
        x[i] = rows / U[i, i]
    return x


def inverse(A: np.ndarray):
    L, U = LU(A)
    dim = A.shape[0]
    I = np.eye(dim, dtype=A.dtype)

    for i in range(dim):
        # 对逆按列分块求解
        I[:, i:i + 1] = solve(L, U, I[:, i:i + 1])
    return I


A = np.array(
    [
        31., -13, 0, 0, 0, -10, 0, 0, 0,
        -13, 35, -9, 0, -11, 0, 0, 0, 0,
        0, -9, 31, -10, 0, 0, 0, 0, 0,
        0, 0, -10, 79, -30, 0, 0, 0, -9,
        0, 0, 0, -30, 57, -7, 0, -5, 0,
        0, 0, 0, 0, -7, 47, -30, 0, 0,
        0, 0, 0, 0, 0, -30, 41, 0, 0,
        0, 0, 0, 0, -5, 0, 0, 27, -2,
        0, 0, 0, -9, 0, 0, 0, -2, 29
    ]
).reshape((9, 9))
b = np.array([-15., 27, -23, 0, -20, 12, -7, 7, 10]).reshape((9, 1))
L, U = LU(A)
x = solve(L, U, b)
print('LU:\n', x, det(U))
P, L, U = PLU(A)
x = solve(L, U, np.matmul(P, b))
print('列主元LU:\n')
AI = inverse(A)
I = np.matmul(A, AI)
input()
