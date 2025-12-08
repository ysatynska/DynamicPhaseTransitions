import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------- Parameters from the MFGL free energy ----------
# F(m) = a m^2 + b m^4 - h m
b = 3 * np.sqrt(3) / 4 # material-specific constant
h = 0.0

# Several 'a' values that illustrate the change in the potential
a_values = [1.5, 0.0, -1.5]

m_min, m_max = -2.0, 2.0
m = np.linspace(m_min, m_max, 1000)

def F(m, a, b, h):
    return a * m**2 + b * m**4 - h * m

plt.figure(figsize=(8, 5))

colors = ["red", "darkgreen", "blue"]
linestyles = ["-", "--", ":"]

for a, c, ls in zip(a_values, colors, linestyles):
    F_vals = F(m, a, b, h)
    label = fr"$a = {a}$"
    plt.plot(m, F_vals, color = c, linestyle=ls, linewidth=3, label=label)

plt.xlim(-2.0, 2.0)
plt.ylim(-1.0, 1.0)

plt.xlabel(r"$m(t)$", fontsize=20)
plt.ylabel(r"$F(m)$", fontsize=20)
ax = plt.gca()

# major ticks every 0.5 on both axes
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

# thicker axes borders (spines)
for spine in ax.spines.values():
    spine.set_linewidth(1.3)
    
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("figures/Figure1.png", dpi=300)
plt.show()
