import numpy as np
import matplotlib.pyplot as plt

from functions.bisection import bisection
from functions.fixed_point import fixed_point

# ===============================
# CHANGE THIS FUNCTION
# ===============================

def f(x):
    return np.sqrt(x) - np.cos(x)

# Fixed point version
def g(x):
    return x - f(x)

a = 0
b = 1
x0 = 0.5


# ------------------------------
# Test Functions
# ------------------------------
# def f(x): return np.cos(x) - x
# a, b = 0, 1
# x0 = 0.5

# def f(x): return np.exp(-x) - x
# a, b = 0, 1
# x0 = 0.5

# def f(x): return np.sqrt(x) - np.cos(x)
# a, b = 0, 1
# x0 = 0.5

# ===============================
# RUN METHODS
# ===============================

root_bis, hist_bis = bisection(f, a, b)
root_fp, hist_fp = fixed_point(g, x0)

print("Bisection Root:", root_bis)
print("Fixed Point Root:", root_fp)

# ===============================
# COMMON FUNCTION PLOT DATA
# ===============================

x_vals = np.linspace(-0.1, 1, 400)
y_vals = f(x_vals)

# ===============================
# GRAPH 1: BISECTION
# ===============================

plt.figure()
plt.plot(x_vals, y_vals, label="f(x)")
plt.axhline(0)

plt.scatter(hist_bis, [f(h) for h in hist_bis],
            label="Iterations")

plt.scatter(root_bis, f(root_bis),
            label="Root")

plt.title("Bisection Method")
plt.legend()
plt.grid()

# ===============================
# GRAPH 2: FIXED POINT
# ===============================

plt.figure()
plt.plot(x_vals, y_vals, label="f(x)")
plt.axhline(0)

plt.scatter(hist_fp, [f(h) for h in hist_fp],
            label="Iterations")

plt.scatter(root_fp, f(root_fp),
            label="Root")

plt.title("Fixed Point Method")
plt.legend()
plt.grid()

plt.show()