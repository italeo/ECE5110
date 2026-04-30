import numpy as np

from functions.linear_systems import gaussian_elimination, lu_decomposition
from functions.nonlinear_systems import newton_system

# LINEAR SYSTEM TEST
# -------------------------------

A = np.array([[3, 2, -4],
              [2, 3, 3],
              [5, -3, 1]])

b = np.array([3, 15, 14])

x = gaussian_elimination(A.copy(), b.copy())
print("Gaussian Elimination Solution:", x)

# Check
print("Check Ax:", A @ x)

# LU DECOMPOSITION TEST
# -------------------------------

L, U = lu_decomposition(A)
print("\nL:\n", L)
print("\nU:\n", U)
print("\nCheck LU:\n", L @ U)

# NONLINEAR SYSTEM TEST
# -------------------------------

def F(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,
        x[0] - x[1] - 1
    ])

x0 = np.array([1.0, 1.0])

solution = newton_system(F, x0)

print("\nNonlinear System Solution:", solution)
print("Check F(x):", F(solution))