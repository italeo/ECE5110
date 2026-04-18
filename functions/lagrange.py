# lagrange.py

import numpy as np

def lagrange_interpolation(x, y):
    """
    Construct the Lagrange interpolating polynomial.

    Parameters
    ----------
    x : array-like
        x data points
    y : array-like
        y data points

    Returns
    -------
    numpy.ndarray
        Coefficients of interpolating polynomial
        (highest degree first)
    """

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    n = len(x)
    P = np.zeros(n)

    for k in range(n):

        # Start with polynomial "1"
        Lk = np.array([1.0])
        denom = 1.0

        for i in range(n):
            if i != k:
                # Multiply polynomial by (x - x_i)
                Lk = np.convolve(Lk, np.array([1.0, -x[i]]))
                denom *= (x[k] - x[i])

        Lk = Lk / denom
        P = P + y[k] * Lk

    return P