import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

from functions.lsoe_solver import function_solve_lsoe
from functions.nlsoe_solver import function_solve_nlsoe
from lib.fermentation_model import yeast_data, F_factory, J


def solve_with_history(F, J, x0, tol=1e-6, max_iter=50):
    x = x0.astype(float)
    history = []

    for i in range(max_iter):
        Fx = F(x)
        err = np.linalg.norm(Fx)
        history.append(err)

        Jx = J(x)
        delta = function_solve_lsoe(Jx, -Fx)

        x = x + delta

        if err < tol:
            history.append(err)
            return x, history

    return x, history


def plot_convergence(history):
    plt.figure()
    plt.plot(range(len(history)), history, marker='o')
    plt.yscale('log')
    plt.ylim(1e-10, 1e2)
    plt.xlabel("Iteration")
    plt.ylabel("||F(x)||")
    plt.title("Newton Method Convergence")
    plt.grid(True)


def plot_flux_comparison(results):
    labels = list(results.keys())

    BF_vals = [results[k][1] for k in labels]
    RF_vals = [results[k][2] for k in labels]
    FF_vals = [results[k][3] for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure()

    plt.bar(x - width, BF_vals, width, label='Growth (BF)')
    plt.bar(x, RF_vals, width, label='Respiration (RF)')
    plt.bar(x + width, FF_vals, width, label='Fermentation (FF)')

    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Flux Value")
    plt.title("Flux Comparison Across Yeasts")
    plt.legend()


if __name__ == "__main__":
    x0 = np.array([20.0, 2.0, 2.0, 5.0])

    results = {}

    # Solve for each yeast
    for name, data in yeast_data.items():
        F = F_factory(data)
        solution, history = solve_with_history(F, J, x0)

        print(f"\n{name}")
        print("Solution:", solution)

        results[name] = solution

    # Plot ONLY one convergence (same behavior)
    plot_convergence(history)

    # Plot comparison
    plot_flux_comparison(results)

    plt.show()