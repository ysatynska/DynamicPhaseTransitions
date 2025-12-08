from sympy import symbols, integrate, solve, Eq
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp

h1 = 1.5
P = 100
max_time = 200
time_step = 1
m0 = [0.8]

a = -3 * math.sqrt(3) / 4
b =  3 * math.sqrt(3) / 8

t_vals = np.arange(1, max_time, time_step)
h = h1 * np.cos(2 * np.pi * t_vals / P)

def dm_dt(t, m):
    return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P)

sol = solve_ivp(dm_dt, (0, max_time), m0, t_eval=t_vals)

# ---------- main hysteresis loop, styled like Figure 2 ----------
plt.figure(figsize=(8, 5))
plt.plot(h, sol.y[0], color="green", linewidth=3)

plt.xlabel(r"$h(t)$", fontsize=20)
plt.ylabel(r"$m(t)$", fontsize=20)

ax = plt.gca()

# major ticks every 0.5 on both axes
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# thicker axes borders (spines)
for spine in ax.spines.values():
    spine.set_linewidth(1.3)

plt.tight_layout()
plt.savefig("figures/Figure3a.png", dpi=300)
plt.show()
