import numpy as np
from functions.nlsoe_solver import function_solve_nlsoe
from lib.fermentation_model import yeast_data, F_factory, J_factory

def test_nlsoe():
    x0 = np.array([10.0, 2.0, 2.0, 5.0])

    for name, data in yeast_data.items():
        F = F_factory(data)
        J = J_factory(data)

        sol = function_solve_nlsoe(F, J, x0)

        residual = np.linalg.norm(F(sol))

        print(f"\n{name}")
        print("Solution:", sol)
        print("Residual:", residual)

        assert residual < 1e-6

if __name__ == "__main__":
    test_nlsoe()