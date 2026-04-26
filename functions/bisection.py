import numpy as np

def bisection(f, a, b, err=1e-6, max_steps=100):
    """
    Bisection method based on ECE5110 slides
    """

    sol_min = a
    sol_max = b

    if f(sol_min) * f(sol_max) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    step = 0
    history = []

    while step < max_steps:
        # Step 2: midpoint
        mid = sol_min / 2 + sol_max / 2
        history.append(mid)

        # Step 3: check error
        if abs(f(mid)) < err:
            return mid, history

        # Step 4: decide interval
        if f(sol_min) * f(mid) < 0:
            sol_max = mid
        else:
            # Step 5
            sol_min = mid

        step += 1

    print("Warning: Max steps reached")
    return mid, history