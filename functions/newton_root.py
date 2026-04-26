def newton_root(f, df, x0, tol=1e-6, max_iter=50):
    x = x0
    history = [x0]

    for _ in range(max_iter):
        dfx = df(x)

        if dfx == 0:
            print("Zero derivative encountered")
            return x, history

        x_new = x - f(x)/df(x)
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, history

        x = x_new

    print("Warning: did not converge")
    return x, history