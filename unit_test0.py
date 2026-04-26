import numpy as np
import matplotlib.pyplot as plt

from functions.bisection import bisection
from functions.fixed_point import fixed_point
from functions.newton_root import newton_root

# ===============================
# CHANGE THIS FUNCTION
# ===============================

def f(x):
    return np.sqrt(x) - np.cos(x)

# Derivative for Newton
def df(x):
    return 1/(2*np.sqrt(x)) + np.sin(x)

# Fixed point version
def g(x):
    return x - f(x)

a = 0
b = 1
x0 = 0.5

# ===============================
# RUN METHODS
# ===============================

root_bis, hist_bis = bisection(f, a, b)
root_fp, hist_fp = fixed_point(g, x0)
root_newton, hist_newton = newton_root(f, df, x0)

print("Bisection Root:", root_bis)
print("Fixed Point Root:", root_fp)
print("Newton Root:", root_newton)

# ===============================
# COMMON FUNCTION PLOT DATA
# ===============================

x_vals = np.linspace(0.001, 1, 400)  # avoid sqrt(0) issues
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

# ===============================
# GRAPH 3: NEWTON METHOD
# ===============================

plt.figure()
plt.plot(x_vals, y_vals, label="f(x)")
plt.axhline(0)

plt.scatter(hist_newton, [f(h) for h in hist_newton],
            color='purple', label="Iterations")

plt.scatter(root_newton, f(root_newton),
            color='red', label="Root")

plt.title("Newton Method")
plt.legend()
plt.grid()

plt.show()