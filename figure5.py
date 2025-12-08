# import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import odeint
from scipy.optimize import fsolve

# set number of digits for mpmath arbitrary precision
mp.dps = 20


def np_tdgl_deriv(m1,t1):   # define the TDGL derivative with a sinusoidal applied field (numpy version)
    m = m1
    dmdt = -2*np_a*m - 4*np_b*m**3 + np_h0*np.cos(2*np.pi*t1/np_P)
    return dmdt
    
def mp_tdgl_deriv(t1,m1):   # define the TDGL derivative with a sinusoidal applied field (mpmath version)
    m = m1
    dmdt = -2*mp_a*m - 4*mp_b*m**3 + mp_h0*mp.cos(2*np.pi*t1/mp_P)
    return dmdt

def np_func(m0):                          # define function which returns difference between initial
    t2 = np.linspace(0,np_P,numdivs)         # magnetization value and the final (integrated) magnetization
    msol= odeint(np_tdgl_deriv,m0,t2)           #  (numpy version)
    diff = msol[-1]-m0
#    print('np_func iteration: m0 diff = ' + str(m0) + ' ' + str(diff))
    return diff 

def mp_func(m0):                          # define function which returns difference between initial
    msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0)       # magnetization value and the final (integrated) magnetization
    diff = msol(mp_P) - m0                                       # (mpmath version)
#    print('mp_func iteration: m0 diff  = ' + str(m0) + ' ' + str(diff))
#    print(".",end="")
    return diff

# set parameters of TDGL model for mpmath and numpy, as well as critical period found in another program
mp_a = -mp.mpf(3.0)*mp.sqrt(mp.mpf(3.0))/mp.mpf(4.0)
mp_b = mp.mpf(3.0)*mp.sqrt(mp.mpf(3.0))/mp.mpf(8.0)
mp_h0 = mp.mpf(1.5)
np_a = -3.0 * np.sqrt(3.0)/4.0
np_b = 3.0 * np.sqrt(3.0)/8.0
np_h0 = 1.5
P_c_np = 5.31935766199729
P_c_mp = mp.mpf(5.31935766199729)
numdivs = 10000

# create list of epsilon and period values to use in the mn_mag_scaling
eps_list_log10_np = np.linspace(-9.0,-7.0,5)
eps_list_len = len(eps_list_log10_np)
eps_list_10_np = np.linspace(10.,10.,eps_list_len)
eps_list_log10_mp = mp.linspace(mp.mpf(-9.0),mp.mpf(-7.0),5)
eps_list_10_mp = mp.linspace(mp.mpf(10.),mp.mpf(10.),eps_list_len)
P_list_np = P_c_np * (1.0 - np.power(eps_list_10_np,eps_list_log10_np))
P_list_mp = [P_c_mp * (mp.mpf(1.0) - mp.power(eps_list_10_mp[i],eps_list_log10_mp[i])) for i in range(0,eps_list_len)]
eps_list_mp = [(P_c_mp - P_list_mp[i])/P_c_mp for i in range(0,eps_list_len)]
eps_sqrt_list_mp = [mp.sqrt(i) for i in eps_list_mp]

# specify number of mn_mag fourier components to calculate
num_fourier_comps = 8

# define functions used as integrands in calculating to Fourier components
def m0_integrand(t):
     return msol(t)
def mn_cos_integrand(t):
     return msol(t)*mp.cos(mp.mpf(2.0)*mp.pi*N*t/mp_P)
def mn_sin_integrand(t):
     return msol(t)*mp.sin(mp.mpf(2.0)*mp.pi*N*t/mp_P)

# create array to hold the fourier components computed at the critical period
mn2_mag_array_pc = np.array([mp.mpf(0) for i in range(num_fourier_comps)])

np_P = P_c_np
mp_P = P_c_mp
p3 = mp_P
m0_init = -0.55
print('P = ',p3)
m0_soln_np = fsolve(np_func,m0_init)
print('m0_soln_np = ',m0_soln_np)
m0_soln_mp = mp.findroot(mp_func,[0.98*m0_soln_np[0],1.02*m0_soln_np[0]],tol=1.0e-10,solver = 'bisect')
print('m0_soln_mp = ',m0_soln_mp)
msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0_soln_mp)

m0 = (mp.mpf(1.0)/mp_P)*mp.quadsubdiv(m0_integrand, [0,mp_P])
m0_complex = m0 + 1j * 0
mn2_mag_array_pc[0] = m0*m0

for N in range(1,num_fourier_comps):
    mn_cos = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_cos_integrand,[0,mp_P])
    mn_sin = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_sin_integrand,[0,mp_P])
    mn_complex = 0.5*(mn_cos - 1j * mn_sin)
    mn_complex_conj = 0.5*(mn_cos + 1j * mn_sin)
    mn_mag2 = mn_complex*mn_complex_conj
    mn2_mag_array_pc[N] = mp.re(mn_mag2)

epsilon = (P_c_mp - mp_P)/P_c_mp
print('epsilon = (P_c = P)/P_c: ',epsilon)
print('mn2_mag_array_pc = ',mn2_mag_array_pc)
print()

rows = num_fourier_comps
cols = eps_list_len
mn2_mag_array = np.array([[mp.mpf(0) for i in range(cols)] for j in range(rows)])
zn2_mag_array = np.array([[mp.mpf(0) for i in range(cols)] for j in range(rows)])

for i in range(0,eps_list_len):
    np_P = P_list_np[i]
    mp_P = P_list_mp[i]
    p3 = mp_P
    m0_init = -0.55
    print('P = ',p3)
    m0_soln_np = fsolve(np_func,m0_init)
    print('m0_soln_np = ',m0_soln_np)
    m0_soln_mp = mp.findroot(mp_func,[0.98*m0_soln_np[0],1.02*m0_soln_np[0]],tol=1.0e-10,solver = 'bisect')
    print('m0_soln_mp = ',m0_soln_mp)
    msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0_soln_mp)

    m0 = (mp.mpf(1.0)/mp_P)*mp.quadsubdiv(m0_integrand, [0,mp_P])
    mn2_mag_array[0,i] = m0*m0
    zn2_mag_array[0,i] = mn2_mag_array[0,i] - mn2_mag_array_pc[0]
    if zn2_mag_array[0,i] < 0.0:
        zn2_mag_array[0,i] = -zn2_mag_array[0,i]
    
    for N in range(1,num_fourier_comps):
        mn_cos = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_cos_integrand,[0,mp_P])
        mn_sin = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_sin_integrand,[0,mp_P])
        mn_complex = 0.5*(mn_cos - 1j * mn_sin)
        mn_complex_conj = 0.5*(mn_cos + 1j * mn_sin)
        mn_mag2 = mn_complex*mn_complex_conj
        print(str(N)+' '+str(i)+' '+str(mn_mag2))
        mn2_mag_array[N,i] = mp.re(mn_mag2)
        zn2_mag_array[N,i] = mn2_mag_array[N,i] - mn2_mag_array_pc[N]
        if zn2_mag_array[N,i] < 0.0:
            zn2_mag_array[N,i] = -zn2_mag_array[N,i]
            
    epsilon = (P_c_mp - mp_P)/P_c_mp
    print('i = ',i)
    print('epsilon = (P_c = P)/P_c: ',epsilon)
    print('mn2_mag_array = ',mn2_mag_array[:,i])
    print('zn2_mag_array = ',zn2_mag_array[:,i])
    print()

# ------------------ Styled plotting block ------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Convert eps_list and z data to floats for plotting
eps_vals = np.array([float(e) for e in eps_list_mp])

# log–log axes
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r"$\varepsilon = \dfrac{P_c - P}{P_c}$", fontsize=20)
ax.set_ylabel(r"$z_k$", fontsize=20)

# Reference scaling line ~ ε^{1/2}
ref_line = eps_vals**0.5
ax.plot(eps_vals, ref_line, 'k--', lw=2, label=r"$\varepsilon^{1/2}$")

# z_k curves (zn2_mag_array stores z_k^2, so take sqrt)
for loop in range(num_fourier_comps):
    z_vals = [float(mp.sqrt(val)) for val in zn2_mag_array[loop, :]]
    ax.plot(eps_vals, z_vals, lw=2, label=fr"$z_{{{loop}}}$")

for spine in ax.spines.values():
    spine.set_linewidth(1.3)

ax.tick_params(axis='both', which='both', labelsize=14)
ax.legend(loc="upper left", fontsize=15, frameon=True)

plt.tight_layout()
plt.show()
plt.savefig("figures/Figure5.png", dpi=300)
plt.close()
