import numpy as np

def least_squares(x, y):
    n = len(x)

    x = np.array(x)
    y = np.array(y)

    a = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - (np.sum(x))**2)
    b = np.mean(y) - a*np.mean(x)

    return a, b