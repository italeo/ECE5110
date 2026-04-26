import numpy as np

def jacobian(F, x, h=1e-5):
    n = len(x)
    J = np.zeros((n, n))

    for i in range(n):
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[i] += h
        x_backward[i] -= h

        J[:, i] = (F(x_forward) - F(x_backward)) / (2*h)

    return J


def newton_system(F, x0, tol=1e-6, max_iter=50):
    x = x0.astype(float)

    for _ in range(max_iter):
        J = jacobian(F, x)
        Fx = F(x)

        delta = np.linalg.solve(J, -Fx)
        x = x + delta

        if np.linalg.norm(delta) < tol:
            return x

    print("Warning: did not converge")
    return x