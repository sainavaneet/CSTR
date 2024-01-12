
from scipy.integrate import solve_ivp
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = 'plotly_dark'
import numpy as np

q = 100.
CAin = 1.
Tin = 350.
V = 100.
p = 1000.
C  = 0.239
HR = 5e4
Ac = 8750.
K0 = 7.2e10
UA = 5e4
Tc0 = 300.
CA0 = 0.5
T0 = 350.
ti = 300
tf = 310


def right(t, v):
    CA,T = v
    K = K0 * np.exp(-Ac/T)
    return [q/V*(CAin-CA)-K*CA,q/V*(Tin-T)+(HR*K*CA)/(p*C)-UA*(T-Tc0)/(p*V*C)]

soln = solve_ivp(right ,(ti ,tf) , [CA0 , T0] , method='Radau' , dense_output=True).sol

tp = np.linspace(ti , tf , 100)
CA , T = soln(tp)



fig = make_subplots(cols=2 , rows = 1)
fig.add_scatter(x = tp , y = CA, mode = 'lines' , name= 'dCA/dt' , row = 1 , col = 1)
fig.add_scatter(x = tp , y = T, mode = 'lines' , name= 'dT/dt' , row = 1 , col = 2)

fig.show()