import numpy as np
from functions.lsoe_solver import function_solve_lsoe
from functions.jacobian_fd import compute_jacobian_fd


def function_solve_nlsoe(F, J, x0, tol=1e-6, max_iter=50, use_fd=False, verbose=True):
    """
    Newton method for nonlinear systems
    """

    x = x0.astype(float)

    for i in range(max_iter):
        Fx = F(x)

        if use_fd:
            Jx = compute_jacobian_fd(F, x)
        else:
            Jx = J(x)

        try:
            delta = function_solve_lsoe(Jx, -Fx)
        except:
            raise ValueError("Jacobian is singular")

        x = x + delta

        err = np.linalg.norm(Fx)

        if verbose:
            print(f"Iter {i+1}: ||F(x)|| = {err:.6e}, x = {x}")

        if err < tol:
            print(f"\n Converged in {i+1} iterations")
            return x

    raise ValueError(" Newton method did not converge")