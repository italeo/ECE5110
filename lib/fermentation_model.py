import numpy as np

# Data from paper
growth_rate = 2.3
ethanol_rate = 5.4
co2_rate = 6.7


def F(x):
    """
    x = [GF, BF, RF, FF]
    """

    GF, BF, RF, FF = x

    return np.array([
        GF - BF - RF - FF,
        BF - growth_rate,
        FF - ethanol_rate,
        RF + FF - co2_rate
    ])


def J(x):
    """
    Analytical Jacobian (constant)
    """

    return np.array([
        [1, -1, -1, -1],
        [0,  1,  0,  0],
        [0,  0,  0,  1],
        [0,  0,  1,  1]
    ])