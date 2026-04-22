# rk4_baseball.py

import numpy as np

def rk4(f, t0, y0, h, n):
    """
    Generic RK4 solver

    Parameters:
    f  : function f(t, y)
    t0 : initial time
    y0 : initial state (numpy array)
    h  : step size
    n  : number of steps

    Returns:
    t_values, y_values
    """
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0

    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)

        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_values.append(t)
        y_values.append(y)

        # Stop if ball hits ground
        if y[1] < 0:
            break

    return np.array(t_values), np.array(y_values)


def baseball_ode(t, state, params):
    """
    Baseball dynamics with drag + Magnus effect

    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state

    g = params["g"]
    k = params["drag"]
    c = params["magnus"]
    omega = params["spin"]

    v = np.sqrt(vx**2 + vy**2)

    dxdt = vx
    dydt = vy

    dvxdt = -k * v * vx + c * omega * vy
    dvydt = -g - k * v * vy - c * omega * vx

    return np.array([dxdt, dydt, dvxdt, dvydt])


def simulate_pitch(v0, angle_deg, params, h=0.01, t_max=10, initial_state=None):

    if initial_state is not None:
        y0 = np.array(initial_state)
    else:
        angle = np.radians(angle_deg)
        vx0 = v0 * np.cos(angle)
        vy0 = v0 * np.sin(angle)
        y0 = np.array([0.0, 0.0, vx0, vy0])

    def f(t, y):
        return baseball_ode(t, y, params)

    n_steps = int(t_max / h)
    t_vals, y_vals = rk4(f, 0, y0, h, n_steps)

    return t_vals, y_vals