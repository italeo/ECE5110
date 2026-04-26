import numpy as np

def cubic_spline(x, y, x_eval):
    n = len(x)
    a = y.copy()

    h = np.diff(x)

    alpha = np.zeros(n)
    for i in range(1, n-1):
        alpha[i] = (3/h[i])*(a[i+1] - a[i]) - (3/h[i-1])*(a[i] - a[i-1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n-1):
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]

    c = np.zeros(n)
    b = np.zeros(n-1)
    d = np.zeros(n-1)

    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (a[j+1] - a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3
        d[j] = (c[j+1] - c[j])/(3*h[j])

    # find interval
    for i in range(n-1):
        if x_eval >= x[i] and x_eval <= x[i+1]:
            dx = x_eval - x[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3