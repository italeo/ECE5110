import numpy as np
import matplotlib.pyplot as plt

from functions.rk1 import euler
from functions.rk4 import rk4

# TEST ODE
# -------------------------------

# dy/dx = y
def f(x, y):
    return y

# exact solution
def y_exact(x):
    return np.exp(x)

# PARAMETERS
# -------------------------------

x0 = 0
y0 = 1
h = 0.1
n = 20

# RUN METHODS
# -------------------------------

x_euler, y_euler = euler(f, x0, y0, h, n)
x_rk4, y_rk4 = rk4(f, x0, y0, h, n)

x_vals = np.linspace(0, 2, 200)
y_vals = y_exact(x_vals)

# PLOT
# -------------------------------

plt.figure(figsize=(6,4))

plt.plot(x_vals, y_vals, label="Exact (e^x)", color='black')
plt.plot(x_euler, y_euler, 'o-', label="Euler (RK1)", color='blue')
plt.plot(x_rk4, y_rk4, 's-', label="RK4", color='red')

plt.title("ODE Solution Comparison")
plt.legend()
plt.grid()

plt.show()