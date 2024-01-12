import numpy
import scipy
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt


q = 100  # L/min
cA_i = 1  # mol/L
T_i = 350  # K
V = 100  # L
rho = 1000 # g/L
C = 0.239 # J/(g K)
Hr = -5e4  # J/(g K)
E_over_R = 8750  # K
k0 = 7.2e10  # 1/min
UA = 5e4  # J/(min K)
Tc = Tc0 = 300  # K

cA0 = 0.5  # mol/L
T0 = 350  # K

def intsys(t, x):
    cA, T = x
    k = k0*numpy.exp(-E_over_R/T)
    w = q*rho
    dcAdt = q*(cA_i - cA)/V - k*cA
    dTdt = 1/(V*rho*C)*(w*C*(T_i - T) - Hr*V*k*cA + UA*(Tc - T))
    return dcAdt, dTdt

x0 = [cA0, T0]

def ss(x):
    """ This wrapper function simply calls intsys with a zero time"""
    return intsys(0, x)

x0 = scipy.optimize.fsolve(ss, x0)

tspan = (0, 10)

ss(x0)
t = numpy.linspace(*tspan, 1000)
def simulate():
    r = scipy.integrate.solve_ivp(intsys, tspan, x0, t_eval=t)
    return r.y

cA, T = simulate()
plt.plot(t, T)
plt.ylim(349, 351)
plt.show()

fig, (axT, axcA) = plt.subplots(2, 1, figsize=(7, 8))
for Tc in [290, 300, 305]:
    cA, T = simulate()
    axT.plot(t, T, label='{} K'.format(Tc))
    axT.set_ylabel('Reactor temperature (K)')
    axcA.plot(t, cA, label='{} K'.format(Tc))
    axcA.set_ylabel('Reactant A concentration (mol/L)')
axcA.legend()
plt.show()