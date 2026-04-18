# test_lagrange.py

import unittest
import numpy as np
import matplotlib.pyplot as plt
from functions import lagrange_interpolation


class TestLagrangeSurfTide(unittest.TestCase):

    def test_mid_tide_prediction(self):

        # Time in hours since Feb 24, 2026 00:00
        t = np.array([
            1.92,    # 1:55 AM (High)
            10.28,   # 10:17 AM (Low)
            27.40,   # Feb 25 3:24 AM (High)
            35.67    # Feb 25 11:40 AM (Low)
        ])

        # Corresponding tide heights (ft)
        h = np.array([
            4.9,
            0.3,
            5.0,
            -0.2
        ])

        coeffs = lagrange_interpolation(t, h)

        # Mid tide level
        h_mid = (np.max(h) + np.min(h)) / 2

        # Solve P(t) - h_mid = 0
        coeffs_mid = coeffs.copy()
        coeffs_mid[-1] -= h_mid

        roots = np.roots(coeffs_mid)
        real_roots = roots[np.isreal(roots)].real
        valid_times = [r for r in real_roots if min(t) <= r <= max(t)]

        print("\nOptimal Surf Windows (±30 minutes around mid-tide):\n")

        # Plot preparation
        tt = np.linspace(min(t), max(t), 500)
        P = np.polyval(coeffs, tt)

        plt.figure()
        plt.plot(tt, P, label="Interpolated Tide Curve")
        plt.scatter(t, h, color="red", label="Tide Data Points")
        plt.axhline(h_mid, linestyle="--", label="Mid Tide Level")

        for r in sorted(valid_times):

            # 30 minute band (0.5 hours)
            start = r - 0.5
            end = r + 0.5

            # Convert to readable time
            for label_time, label_name in [(start, "Start"), (end, "End")]:

                day = "Feb 24" if label_time < 24 else "Feb 25"
                hour = label_time % 24
                h_int = int(hour)
                minutes = int((hour - h_int) * 60)

                if label_name == "Start":
                    start_str = f"{day} {h_int:02d}:{minutes:02d}"
                else:
                    end_str = f"{day} {h_int:02d}:{minutes:02d}"

            print(f"Surf Window: {start_str}  →  {end_str}")

            # Shade surf window
            plt.axvspan(start, end, alpha=0.2)

        plt.legend()
        plt.xlabel("Time (hours since Feb 24, 00:00)")
        plt.ylabel("Tide Height (ft)")
        plt.title("El Porto Mid-Tide Surf Prediction (±30 min Window)")
        plt.show()

        # Verify interpolation
        for i in range(len(t)):
            self.assertAlmostEqual(np.polyval(coeffs, t[i]), h[i], places=5)


if __name__ == "__main__":
    unittest.main()