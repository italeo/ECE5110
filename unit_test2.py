import numpy as np
import matplotlib.pyplot as plt

from functions.differentiation import forward_diff, central_diff
from functions.integration import midpoint, trapezoidal

# TEST FUNCTION
# -------------------------------

def f(x):
    return np.sin(x)

def df_exact(x):
    return np.cos(x)

def integral_exact(a, b):
    return -np.cos(b) + np.cos(a)

# ----- Cubic function for testing -----
# def f(x): return x**3 - 2*x + 1
# def df_exact(x): return 3*x**2 - 2
# def integral_exact(a, b): return (b**4/4 - b**2 + b) - (a**4/4 - a**2 + a)

# ----- Exponential function for testing -----
# def f(x): return np.exp(x)
# def df_exact(x): return np.exp(x)
# def integral_exact(a, b): return np.exp(b) - np.exp(a)

# ----- Logarithmic function for testing -----
# def f(x): return np.log(x)
# def df_exact(x): return 1/x
# def integral_exact(a, b): return b*np.log(b) - b - (a*np.log(a) - a)

# DIFFERENTIATION TEST
# -------------------------------

x0 = 1.0
h_vals = np.logspace(-6, -1, 50)

error_forward = []
error_central = []

for h in h_vals:
    fd = forward_diff(f, x0, h)
    cd = central_diff(f, x0, h)
    exact = df_exact(x0)

    error_forward.append(abs(fd - exact))
    error_central.append(abs(cd - exact))


plt.figure(figsize=(6,4))
plt.loglog(h_vals, error_forward, label="Forward Diff Error")
plt.loglog(h_vals, error_central, label="Central Diff Error")

plt.title("Differentiation Error")
plt.xlabel("h")
plt.ylabel("Error")
plt.legend()
plt.grid()


# INTEGRATION TEST
# -------------------------------

a, b = 0, np.pi
n_vals = range(10, 1000, 10)

error_mid = []
error_trap = []

exact = integral_exact(a, b)

for n in n_vals:
    m = midpoint(f, a, b, n)
    t = trapezoidal(f, a, b, n)

    error_mid.append(abs(m - exact))
    error_trap.append(abs(t - exact))


plt.figure(figsize=(6,4))
plt.plot(n_vals, error_mid, label="Midpoint Error")
plt.plot(n_vals, error_trap, label="Trapezoidal Error")

plt.title("Integration Error")
plt.xlabel("n")
plt.ylabel("Error")
plt.legend()
plt.grid()


plt.show()