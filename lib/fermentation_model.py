import numpy as np

# Constants for nonlinear model
k1 = 0.25
k2 = 0.002

yeast_data = {
    "Yeast A (Y706)": {
        "growth_rate": 2.3,
        "ethanol_rate": 5.4,
        "co2_rate": 6.7
    },
    "Yeast B (Y705)": {
        "growth_rate": 2.0,
        "ethanol_rate": 4.9,
        "co2_rate": 6.7
    },
    "Yeast C (Y710)": {
        "growth_rate": 1.8,
        "ethanol_rate": 4.5,
        "co2_rate": 6.7
    }
}


def F_factory(data):
    def F(x):
        GF, BF, RF, FF = x

        return np.array([
            GF - BF - RF - FF,
            BF - k1 * GF * data["growth_rate"],
            FF - k2 * GF**2 * data["ethanol_rate"],
            RF + FF - data["co2_rate"]
        ])
    return F


def J_factory(data):
    def J(x):
        GF, BF, RF, FF = x

        return np.array([
            [1, -1, -1, -1],
            [-k1 * data["growth_rate"], 1, 0, 0],
            [-2 * k2 * GF * data["ethanol_rate"], 0, 0, 1],
            [0, 0, 1, 1]
        ])
    return J