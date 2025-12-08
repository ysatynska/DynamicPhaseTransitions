import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve, root_scalar

mp.dps = 25

mp_a = -mp.mpf(3) * mp.sqrt(3) / 4
mp_b = mp.mpf(3) * mp.sqrt(3) / 8
mp_h1 = mp.mpf('1.5')

np_a = -3.0 * np.sqrt(3.0) / 4.0
np_b =  3.0 * np.sqrt(3.0) / 8.0
np_h1 = 1.5

P_c_mp = mp.mpf('5.31935766199729')
num_k = 4
h_mults = np.logspace(-11, -10, 5)

mp_h0 = mp.mpf('0.3')
mp_h2 = mp.mpf('0.4')
mp_h4 = mp.mpf('0.2')

np_h0 = 0.3
np_h2 = 0.4
np_h4 = 0.2

num_divs = 10000

def get_init_m(period, h_mult=mp.mpf(0)):
    p_np = float(period)
    h_mult_np = float(h_mult)
    m0_guess = -0.55

    def np_tdgl_deriv(t, m):
        return -2 * np_a * m - 4 * np_b * m**3 + np_h1 * np.cos(2 * np.pi * t / p_np) + \
               h_mult_np * (np_h0 + np_h2 * np.cos(2 * 2 * np.pi * t / p_np) +
                            np_h4 * np.cos(4 * 2 * np.pi * t / p_np))

    def mp_tdgl_deriv(t, m):
        return -2 * mp_a * m - 4 * mp_b * m**3 + mp_h1 * mp.cos(2 * np.pi * t / period) + \
               h_mult * (mp_h0 + mp_h2 * mp.cos(2 * 2 * np.pi * t / period) +
                         mp_h4 * mp.cos(4 * 2 * np.pi * t / period))

    def np_func(m0):
        msol = odeint(np_tdgl_deriv, m0, np.linspace(0, p_np, num_divs))
        return msol[-1] - m0

    m0_seed = fsolve(np_func, m0_guess, xtol=1e-6)[0]

    def residue(mp_m0):
        sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), mp_m0)
        return sol(period) - mp_m0

    def f(m):
        return float(residue(m))

    step = 1e-4
    a, b_ = m0_seed - step, m0_seed + step
    fa, fb = f(a), f(b_)

    for _ in range(25):
        if np.sign(fa) == 0:
            return mp.mpf(a)
        if np.sign(fb) == 0:
            return mp.mpf(b_)
        if np.sign(fa) != np.sign(fb):
            break
        step *= 1.6
        a, b_ = m0_seed - step, m0_seed + step
        fa, fb = f(a), f(b_)

    sol = root_scalar(f, bracket=[a, b_], method='brentq', xtol=1e-10)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge")

    return mp.mpf(sol.root)

def get_m_k(m0, period, h_mult=mp.mpf(0)):
    def mp_tdgl_deriv(t, m):
        return -2 * mp_a * m - 4 * mp_b * m**3 + mp_h1 * mp.cos(2 * np.pi * t / period) + \
               h_mult * (mp_h0 + mp_h2 * mp.cos(2 * 2 * np.pi * t / period) +
                         mp_h4 * mp.cos(4 * 2 * np.pi * t / period))

    sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
    m_k_cs = []

    m0_mean = mp.quad(lambda t: sol(t), [0, period]) / period
    m_k_cs.append(m0_mean)

    for k in range(1, num_k):
        omega_k = 2 * np.pi * k / period
        cos_int = mp.quad(lambda t: sol(t) * mp.cos(omega_k * t), [0, period]) / period
        sin_int = mp.quad(lambda t: sol(t) * mp.sin(omega_k * t), [0, period]) / period
        m_k_cs.append(cos_int - 1j * sin_int)

    return np.array(m_k_cs)

# ----- compute reference m_k at h_mult = 0 -----
init_m_c = get_init_m(P_c_mp)
mk_c = get_m_k(init_m_c, P_c_mp)

# ----- compute dm_k for each h_mult -----
delta_mks = []
for h_mult in h_mults:
    print(h_mult)
    h_mult_mp = mp.mpf(h_mult)
    m0_h = get_init_m(P_c_mp, h_mult_mp)
    mk = get_m_k(m0_h, P_c_mp, h_mult_mp)
    delta_mks.append(mk - mk_c)

# ================== PLOT: even n ==================
fig, ax = plt.subplots(figsize=(4, 5))

# log–log axes + style
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='both', which='both', labelsize=14)
for spine in ax.spines.values():
    spine.set_linewidth(1.3)

# ---- custom x ticks: exponents every 0.1 ----
# change -11, -10 to -7, -6 if you really want 10^-7 ... 10^-6.xxx
exp_min, exp_max = -11.0, -10.0
exponents = np.arange(exp_min, exp_max + 0.0001, 0.4)   # -11, -10.9, ..., -10
xticks = 10**exponents
ax.set_xticks(xticks)
ax.set_xticklabels([rf"$10^{{{e:.1f}}}$" for e in exponents])

# reference ~ h_mult^{1/3}
h13 = [float(mp.power(i, mp.mpf(1)/3)) for i in h_mults]
ax.plot(
    h_mults,
    h13,
    'k--',
    lw=2,
    label=r"$h_{\mathrm{mult}}^{1/3}$"
)

# even n components
for n in range(0, num_k, 2):
    re_vals = [float(abs(mp.re(d[n]))) for d in delta_mks]
    ax.plot(
        h_mults,
        re_vals,
        linestyle='-',
        label=rf"$|{{\Re(d m_{n})}}|$"
    )
    if n > 0:
        im_vals = [float(abs(mp.im(d[n]))) for d in delta_mks]
        ax.plot(
            h_mults,
            im_vals,
            linestyle='--',
            label=rf"$|{{\Im(d m_{n})}}|$"
        )
ax.set_ylim(10**-5.4, 10**-3.2)

# exponents -8, -7.8, ..., -6.6, -6.4, -6.2, -6.0?  NO, stop at -6.5
exp_min_y, exp_max_y = -5.4, -3.2
exponents_y = np.arange(exp_min_y, exp_max_y + 1e-6, 0.3)
yticks = 10**exponents_y
ax.set_yticks(yticks)
ax.set_yticklabels([rf"$10^{{{e:.1f}}}$" for e in exponents_y])
ax.set_xlabel(r"$h_{\mathrm{mult}}$", fontsize=20)
ax.set_ylabel(r"$|d m_n|$", fontsize=20)
ax.legend(fontsize=12, loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig("figures/Figure7a.png", dpi=300)
plt.show()
