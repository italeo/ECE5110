def fixed_point(g, x0, err=1e-6, max_steps=100):
    """
    Fixed point iteration based on ECE5110 slides
    """

    x = x0
    history = [x]

    step = 0

    while step < max_steps:
        x_new = g(x)
        history.append(x_new)

        # Step 3 (from slides)
        if abs(x_new - x) < err:
            return x_new, history

        x = x_new
        step += 1

    print("Warning: Max steps reached")
    return x, history