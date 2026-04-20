import numpy as np

# Multiple yeast datasets (from paper-style values)
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
            BF - data["growth_rate"],
            FF - data["ethanol_rate"],
            RF + FF - data["co2_rate"]
        ])
    return F


def J(x):
    return np.array([
        [1, -1, -1, -1],
        [0,  1,  0,  0],
        [0,  0,  0,  1],
        [0,  0,  1,  1]
    ])