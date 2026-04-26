def lagrange(x_data, y_data, x):
    n = len(x_data)
    result = 0

    for k in range(n):
        Lk = 1
        for i in range(n):
            if i != k:
                Lk *= (x - x_data[i]) / (x_data[k] - x_data[i])

        result += y_data[k] * Lk

    return result