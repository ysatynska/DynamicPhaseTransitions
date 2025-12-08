from sympy import symbols, integrate, solve, Eq
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp

h1 = 1.5
P = 1         # period
max_time = 20
time_step = 0.001
m0 = [1.1, -0.9]

a = -3 * math.sqrt(3) / 4
b =  3 * math.sqrt(3) / 8

t_vals = np.arange(0, max_time, time_step)
h = h1 * np.cos(2 * np.pi * t_vals / P)

def dm_dt(t, m):
    return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P)

sol = solve_ivp(dm_dt, (0, max_time), m0, t_eval=t_vals)

# =======================
# 1) Main hysteresis loop
# =======================
plt.figure(figsize=(8, 5))

# first trajectory: light purple
plt.plot(h, sol.y[0], color="red", linewidth=3,
         label=fr"$m(t)$, $m_0 = {m0[0]}$")
# second trajectory: light blue
plt.plot(h, sol.y[1], color="blue", linewidth=3,
         label=fr"$m(t)$, $m_0 = {m0[1]}$")

plt.xlabel(r"$h(t)$", fontsize=22)
plt.ylabel(r"$m(t)$", fontsize=22)
# plt.title(fr"$m$ vs $h$ for Period $P = 1 = 0.188 P_c$",
#           fontsize=15, fontweight="bold")

ax = plt.gca()

# major ticks every 0.5 on both axes
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# thicker axes borders (spines)
for spine in ax.spines.values():
    spine.set_linewidth(1.3)

plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("figures/Figure3b.png", dpi=300)
plt.show()
