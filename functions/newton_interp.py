import numpy as np

def newton_interp(x_data, y_data, x):
    """
    Newton interpolation using divided differences
    """

    n = len(x_data)
    coef = np.copy(y_data).astype(float)

    # Build divided difference table (in-place)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_data[i] - x_data[i - j])

    # Evaluate polynomial using nested multiplication (Horner form)
    result = coef[n - 1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_data[i]) + coef[i]

    return result