import numpy as np

def midpoint(f, a, b, n):
    h = (b - a) / n
    total = 0

    for i in range(n):
        x_mid = a + (i + 0.5)*h
        total += f(x_mid)

    return total * h


def trapezoidal(f, a, b, n):
    h = (b - a) / n
    total = 0.5*(f(a) + f(b))

    for i in range(1, n):
        total += f(a + i*h)

    return total * h