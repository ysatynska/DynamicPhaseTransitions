import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.ticker import MultipleLocator

# --------- Parameters for Figure 2 (Robb et al.) ----------
# F(m) = a m^2 + b m^4 - h m
a = -3 * math.sqrt(3) / 4      # a < 0
b =  3 * math.sqrt(3) / 8      # chosen so minima at m = ±1 when h = 0

# We'll vary h now, keeping a fixed
h_values = [-1.0, -0.5, 0.0]

# --------- m grid ----------
m_min, m_max = -2.0, 2.0
m = np.linspace(m_min, m_max, 1000)

def F(m, a, b, h):
    return a * m**2 + b * m**4 - h * m

# --------- Styled like your Figure 1 / hysteresis plots ----------
plt.figure(figsize=(8, 5))

colors = ["red", "darkgreen", "blue"]
linestyles = ["-", "--", ":"]

for h_val, c, ls in zip(h_values, colors, linestyles):
    F_vals = F(m, a, b, h_val)
    label  = fr"$h = {h_val}$"
    plt.plot(m, F_vals, color=c, linestyle=ls, linewidth=3, label=label)

plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 1.0)

plt.xlabel(r"$m$", fontsize=22)
plt.ylabel(r"$F(m)$", fontsize=22)
ax = plt.gca()
ax.tick_params(axis="both", which="major", labelsize=16)

# major ticks every 0.5 on both axes
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# thicker axes borders (spines)
for spine in ax.spines.values():
    spine.set_linewidth(1.3)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("figures/Figure2.png", dpi=300)
plt.show()
