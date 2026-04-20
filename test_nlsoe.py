import numpy as np
from functions.nlsoe_solver import function_solve_nlsoe
from lib.fermentation_model import F, J


def test_nlsoe():
    x0 = np.array([20.0, 2.0, 2.0, 5.0])

    sol = function_solve_nlsoe(F, J, x0, use_fd=True)

    print("Solution:", sol)

    residual = np.linalg.norm(F(sol))
    print("Residual:", residual)

    assert residual < 1e-6


if __name__ == "__main__":
    test_nlsoe()