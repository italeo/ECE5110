import numpy as np
import matplotlib.pyplot as plt
from functions.newton import newton_method

# Physical constants
g = 9.81
v = 25.0
R_target = 60.0

def f(theta):
    return (v**2 / g) * np.sin(2 * theta) - R_target

def df(theta):
    return (v**2 / g) * 2 * np.cos(2 * theta)

theta0 = 0.5

theta_root, history = newton_method(f, df, theta0)

print(f"Takeoff angle (degrees): {np.degrees(theta_root):.2f}")

# Plot
theta_vals = np.linspace(0, np.pi / 2, 400)
plt.plot(theta_vals, f(theta_vals))
plt.scatter(history, f(np.array(history)), color='red')
plt.axhline(0, linestyle='--', color='k')
plt.xlabel("Theta (radians)")
plt.ylabel("f(theta)")
plt.title("Newton's Method – Ski Jump Takeoff Angle")
plt.show()
