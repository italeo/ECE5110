import numpy as np

def newton_method(f, df, x0, tol=1e-8, max_iter=50):
    """
    Newton's Method for solving f(x) = 0.
    """
    x = x0
    history = [x0]

    for k in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            raise ValueError("Derivative too small; Newton method fails.")

        x_new = x - fx / dfx
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, history

        x = x_new

    raise RuntimeError("Newton method did not converge.")
