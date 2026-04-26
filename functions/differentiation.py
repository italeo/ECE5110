def forward_diff(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2*h)


def second_derivative(f, x, h=1e-5):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)