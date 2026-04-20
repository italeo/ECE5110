import numpy as np

def function_solve_lsoe(A, B):
    """
    Gaussian elimination solver (converted from MATLAB)
    """

    A = A.astype(float).copy()
    B = B.astype(float).copy()

    n = A.shape[0]
    sol = np.zeros(n)

    if n != B.shape[0] or n != A.shape[1]:
        raise ValueError("Matrix dimensions must agree")

    # Forward elimination
    for i in range(n - 1):
        pivot = np.argmax(np.abs(A[i:, i])) + i

        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            B[[i, pivot]] = B[[pivot, i]]

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            B[j] -= factor * B[i]

    # Back substitution
    for i in range(n - 1, -1, -1):
        sol[i] = (B[i] - np.dot(A[i, i + 1:], sol[i + 1:])) / A[i, i]

    return sol