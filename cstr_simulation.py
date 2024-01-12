from casadi import MX, DM, vertcat, Function, exp, integrator
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

q = 100.
CAin = 1.
Tin = 350.
V = 100.
p = 1000.
C = 0.239
HR = 5e4
Ac = 8750.
K0 = 7.2e10
UA = 5e4
Tc0 = 300.
ti = 300.
tf = 310.

CA = MX.sym('CA')
T = MX.sym('T')
states = vertcat(CA, T)

K = K0 * exp(-Ac/T)
rhs = vertcat(q/V*(CAin-CA)-K*CA, q/V*(Tin-T)+(HR*K*CA)/(p*C)-UA*(T-Tc0)/(p*V*C))

ode = {'x':states, 'ode':rhs}
opts = {'tf':tf-ti}
I = integrator('I', 'cvodes', ode, opts)

N = 100
time_vector = np.linspace(ti, tf, N)
CA0 = 0.5
T0 = 350.
X0 = DM([CA0, T0])
R = I(x0=X0)
grid = np.linspace(ti, tf, N)

CA_values = np.zeros(N)
T_values = np.zeros(N)
dCA_dt_values = np.zeros(N)
dT_dt_values = np.zeros(N)

for i in range(N-1):
    step_opts = {'t0': time_vector[i], 'tf': time_vector[i+1]}
    I_step = integrator('I_step', 'cvodes', ode, step_opts)
    R = I_step(x0=X0, p=[])
    X0 = R['xf']
    CA_values[i], T_values[i] = X0.elements()
    dCA_dt_values[i] = q/V*(CAin-CA_values[i])-K0*np.exp(-Ac/T_values[i])*CA_values[i]
    dT_dt_values[i] = q/V*(Tin-T_values[i])+(HR*K0*np.exp(-Ac/T_values[i])*CA_values[i])/(p*C)-UA*(T_values[i]-Tc0)/(p*V*C)

fig = make_subplots(rows=1, cols=2, subplot_titles=('dCA/dt vs Time', 'dT/dt vs Time'))

fig.add_trace(go.Scatter(x=time_vector, y=dCA_dt_values, mode='lines', name='dCA/dt'), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vector, y=dT_dt_values, mode='lines', name='dT/dt'), row=1, col=2)

fig.update_layout(title_text='System Dynamics Over Time')

fig.show()
