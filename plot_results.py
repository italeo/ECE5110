import matplotlib
matplotlib.use('TkAgg')  # Ensure plots show on Linux

import numpy as np
import matplotlib.pyplot as plt

from functions.lsoe_solver import function_solve_lsoe
from lib.fermentation_model import F, J


def solve_with_history(F, J, x0, tol=1e-6, max_iter=50):
    x = x0.astype(float)
    history = []

    for i in range(max_iter):
        Fx = F(x)
        err = np.linalg.norm(Fx)
        history.append(err)

        Jx = J(x)

        # Use YOUR linear solver (assignment requirement)
        delta = function_solve_lsoe(Jx, -Fx)

        x = x + delta

        if err < tol:
            # append final error again for visibility
            history.append(err)
            return x, history

    return x, history


def plot_convergence(history):
    plt.figure()
    plt.plot(range(len(history)), history, marker='o')
    plt.yscale('log')
    plt.ylim(1e-10, 1e2)  # Makes small values visible
    plt.xlabel("Iteration")
    plt.ylabel("||F(x)||")
    plt.title("Newton Method Convergence")
    plt.grid(True)


def plot_fluxes(solution):
    GF, BF, RF, FF = solution

    labels = ["Growth (BF)", "Respiration (RF)", "Fermentation (FF)"]
    values = [BF, RF, FF]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Flux Value")
    plt.title("Yeast Metabolic Flux Distribution")


if __name__ == "__main__":
    x0 = np.array([20.0, 2.0, 2.0, 5.0])

    solution, history = solve_with_history(F, J, x0)

    print("Solution:", solution)
    print("History:", history)

    plot_convergence(history)
    plot_fluxes(solution)

    # Show BOTH plots at once
    plt.show()