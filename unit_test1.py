import numpy as np
import matplotlib.pyplot as plt

from functions.least_squares import least_squares
from functions.lagrange import lagrange
from functions.newton_interp import newton_interp
from functions.cubic_spline import cubic_spline


# ===============================
# CHANGE DATA HERE
# ===============================

x_data = np.array([1, 2, 3, 4])
y_data = np.array([1, 3, 2, 8])

x_vals = np.linspace(1, 4, 200)

# ===============================
# LEAST SQUARES
# ===============================

a, b = least_squares(x_data, y_data)
y_ls = a*x_vals + b

# ===============================
# LAGRANGE
# ===============================

y_lagrange = [lagrange(x_data, y_data, x) for x in x_vals]

# ===============================
# NEWTON
# ===============================

y_newton = [newton_interp(x_data, y_data, x) for x in x_vals]

# ===============================
# SPLINE
# ===============================

y_spline = [cubic_spline(x_data, y_data, x) for x in x_vals]

# ===============================
# PLOT 1: LEAST SQUARES
# ===============================

plt.figure(figsize=(6,4))
plt.scatter(x_data, y_data, color='black', label="Data Points")
plt.plot(x_vals, y_ls, color='blue', label="Least Squares")

plt.title("Least Squares (Trend Line)")
plt.legend()
plt.grid()


# ===============================
# PLOT 2: LAGRANGE
# ===============================

plt.figure(figsize=(6,4))
plt.scatter(x_data, y_data, color='black', label="Data Points")
plt.plot(x_vals, y_lagrange, color='orange', linestyle='--', label="Lagrange")

plt.title("Lagrange Interpolation")
plt.legend()
plt.grid()


# ===============================
# PLOT 3: NEWTON
# ===============================

plt.figure(figsize=(6,4))
plt.scatter(x_data, y_data, color='black', label="Data Points")
plt.plot(x_vals, y_newton, color='green', label="Newton")

plt.title("Newton Interpolation")
plt.legend()
plt.grid()


# ===============================
# PLOT 4: CUBIC SPLINE
# ===============================

plt.figure(figsize=(6,4))
plt.scatter(x_data, y_data, color='black', label="Data Points")
plt.plot(x_vals, y_spline, color='red', label="Cubic Spline")

plt.title("Cubic Spline Interpolation")
plt.legend()
plt.grid()


plt.show()