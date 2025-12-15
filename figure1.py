import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

b = 3 * np.sqrt(3) / 8
h = 0.0

# Several 'a' values that illustrate the change in the potential
a_values = [1.5, 0.0, -3 * np.sqrt(3) / 4]

m_min, m_max = -2.0, 2.0
m = np.linspace(m_min, m_max, 1000)

def F(m, a, b, h):
    return a * m**2 + b * m**4 - h * m

plt.figure(figsize=(8, 5))

colors = ["blue", "green","red"]
linestyles = ["--", "-.", "-"]

for a, c, ls in zip(a_values, colors, linestyles):
    if np.isclose(a, -3 * np.sqrt(3) / 4):
        label = r"$a = -\frac{3\sqrt{3}}{4} \;(\approx -1.299)$"
    else:
        label = fr"$a = {a}$"

    F_vals = F(m, a, b, h)
    plt.plot(m, F_vals, color=c, linestyle=ls, linewidth=3, label=label)

plt.xlim(-2.0, 2.0)
plt.ylim(-1.4, 1.0)

plt.xlabel(r"$m$", fontsize=22)
plt.ylabel(r"$F(m)$", fontsize=22)
ax = plt.gca()
ax.tick_params(axis="both", which="major", labelsize=16)


# very thin vertical lines at m = ±1
ax.axvline(x=1.0, color="gray", linewidth=1, linestyle="--", zorder=0)
ax.axvline(x=-1.0, color="gray", linewidth=1, linestyle="--", zorder=0)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

for spine in ax.spines.values():
    spine.set_linewidth(1.3)

plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("figures/Figure1.png", dpi=300)
plt.show()
