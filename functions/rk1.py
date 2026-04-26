import numpy as np

def euler(f, x0, y0, h, n):
    x_vals = [x0]
    y_vals = [y0]

    x = x0
    y = y0

    for _ in range(n):
        y = y + h * f(x, y)
        x = x + h

        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)