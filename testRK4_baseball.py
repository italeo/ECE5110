# testRK4_baseball.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions.rk4 import simulate_pitch


# =========================
# DATASET GENERATION (ALL PITCHES)
# =========================
def generate_dataset():
    y0 = 1.8

    pitch_configs = {
        "fastball": {"v0": 42, "spin": 0},
        "curveball": {"v0": 34, "spin": 180},
        "slider": {"v0": 38, "spin": 80}
    }

    all_data = []

    for pitch_name, config in pitch_configs.items():

        params = {
            "g": 9.81,
            "drag": 0.003,
            "magnus": 0.0001,
            "spin": config["spin"]
        }

        t_vals, y_vals = simulate_pitch(
            v0=None,
            angle_deg=None,
            params=params,
            initial_state=[0, y0, config["v0"], 0]
        )

        x_true = y_vals[:,0]
        y_true = y_vals[:,1]

        # Add noise
        noise = np.random.normal(0, 0.02, size=len(y_true))
        y_measured = y_true + noise

        df = pd.DataFrame({
            "time": t_vals,
            "x": x_true,
            "y": y_measured,
            "pitch": pitch_name
        })

        all_data.append(df)

    data = pd.concat(all_data)
    data.to_csv("yamamoto_all_pitches.csv", index=False)

    print("Saved dataset: yamamoto_all_pitches.csv")

    return data


# =========================
# VALIDATION PLOTS (OVERLAY)
# =========================
def validation_plot():
    data = generate_dataset()

    y0 = 1.8

    pitch_configs = {
        "fastball": {"v0": 42, "spin": 0},
        "curveball": {"v0": 34, "spin": 180},
        "slider": {"v0": 38, "spin": 80}
    }

    plt.figure(figsize=(10,6))

    for pitch_name, config in pitch_configs.items():

        params = {
            "g": 9.81,
            "drag": 0.003,
            "magnus": 0.0001,
            "spin": config["spin"]
        }

        # RK4 model
        t_rk, y_rk = simulate_pitch(
            v0=None,
            angle_deg=None,
            params=params,
            initial_state=[0, y0, config["v0"], 0]
        )

        # Filter dataset
        pitch_data = data[data["pitch"] == pitch_name]

        # Scatter (data)
        plt.scatter(pitch_data["x"], pitch_data["y"],
                    s=10, label=f"{pitch_name} data")

        # Line (model)
        plt.plot(y_rk[:,0], y_rk[:,1],
                 linewidth=2, label=f"{pitch_name} RK4")

    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Validation: RK4 vs Synthetic Data (All Pitch Types)")
    plt.legend()
    plt.grid()

    plt.show()


# =========================
# ORIGINAL TESTS (UPDATED)
# =========================
def run_tests():
    y0 = 1.8

    pitch_configs = {
        "fastball": {"v0": 42, "spin": 0},
        "curveball": {"v0": 34, "spin": 180},
        "slider": {"v0": 38, "spin": 80}
    }

    plt.figure(figsize=(10,6))

    for pitch_name, config in pitch_configs.items():

        params = {
            "g": 9.81,
            "drag": 0.003,
            "magnus": 0.0001,
            "spin": config["spin"]
        }

        t, y = simulate_pitch(
            v0=None,
            angle_deg=None,
            params=params,
            initial_state=[0, y0, config["v0"], 0]
        )

        plt.plot(y[:,0], y[:,1], label=pitch_name)

    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Baseball Pitch Types (RK4)")
    plt.legend()
    plt.grid()

    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    validation_plot()   # overlay (REQUIRED)
    run_tests()         # clean comparison