import numpy as np

def compute_jacobian_fd(F, x, h=1e-6):
    """
    Finite difference Jacobian (matches lecture notes)
    """

    n = len(x)
    J = np.zeros((n, n))

    Fx = F(x)

    for i in range(n):
        x_h = x.copy()
        x_h[i] += h
        J[:, i] = (F(x_h) - Fx) / h

    return J