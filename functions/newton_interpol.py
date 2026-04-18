# newton_interpol.py

import numpy as np


def newton_interpolation(x, y):
    """
    Construct interpolating polynomial using Newton's divided differences.

    Parameters
    ----------
    x : array-like
        x data points
    y : array-like
        y data points

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients (highest degree first)
    """

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    n = len(x)

    # Step 1: Compute divided differences
    coef = np.copy(y)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])

    # Step 2: Build polynomial in standard form
    P = np.array([0.0])

    for k in range(n):

        # Build Newton basis term
        term = np.array([1.0])

        for i in range(k):
            term = np.convolve(term, np.array([1.0, -x[i]]))

        # Pad arrays to same length
        if len(term) > len(P):
            P = np.pad(P, (len(term) - len(P), 0))
        elif len(P) > len(term):
            term = np.pad(term, (len(P) - len(term), 0))

        P = P + coef[k] * term

    return P