from lib.ece5110tools import ece5110tools
import matplotlib.pyplot as plt
import numpy as np

precision = 1e-2
start = 1
stop = 6
max_steps = 20


def f(x):
    return x*x - 14


tool = ece5110tools() 

fval = tool.get_val(f, 0)

sol_rec, err = tool.solve_bisection_rec(f, start, stop, precision, max_steps)
sol_loop, err = tool.solve_bisection_loop(f, start, stop, precision, max_steps)

xvals = np.linspace(start, stop, 100)
yvals = f(xvals)               #array
yvals2 = [f(x) for x in xvals] #numpy array

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)

plt.plot(xvals, yvals, label="f")
plt.scatter([sol_rec],[f(sol_rec)], c="red", label=f'solution={sol_rec}')
plt.title('Bisection Recursive')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(xvals, yvals, label="f")
plt.scatter([sol_rec],[f(sol_rec)], c="red", label=f'solution={sol_rec}')
plt.title('Bisection Loop')
plt.legend()
plt.grid(True)
plt.show()

print("DONE")


