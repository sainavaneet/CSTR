from casadi import MX, DM, vertcat, integrator, exp, Function, Opti
import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots

# Constants
q = 100.
CAin = 0.08235
Tin = 441.81
V = 100.
p = 1000.
C = 1
HR = 2e5
Ac = 9980.
K0 = 7.2e10
UA = 7e5
Tc0 = 350.
ti = 0.
tf = 10.

# CasADi variables for states
CA = MX.sym('CA')
T = MX.sym('T')
states = vertcat(CA, T)
n_states = states.size1()

# System dynamics
K = K0 * exp(-Ac/T)
rhs = vertcat(q/V*(CAin-CA)-K*CA, q/V*(Tin-T)+(HR*K*CA)/(p*C)-UA*(T-Tc0)/(p*V*C))

# Define a CasADi function for the system dynamics
system_dynamics = Function('sys_dyn', [states], [rhs])

# Set up the integrator
dae = {'x': states, 'ode': rhs}
opts = {'tf': tf-ti}
integrator = integrator('integrator', 'cvodes', dae, opts)

# Initial conditions
x0 = DM([0.5, 350.])

# Optimization problem setup
opti = Opti()
X = opti.variable(n_states)
opti.minimize(X[1])  # Example: Minimize final temperature
opti.subject_to(X == integrator(x0=x0)['xf'])  # System dynamics constraint
opti.set_initial(X, x0.full())

# Solve the optimization problem
p_opts = {'expand': True}
s_opts = {'max_iter': 100}
opti.solver('ipopt', p_opts, s_opts)
sol = opti.solve()

# Extract solution
opt_CA, opt_T = sol.value(X).flatten()

# Time vector for simulation
time_vector = np.linspace(ti, tf, 100)  # 100 points between ti and tf

# Initialize arrays to store the derivative values
dCA_dt_values = np.zeros_like(time_vector)
dT_dt_values = np.zeros_like(time_vector)

# Compute derivatives at each time point
for i, t in enumerate(time_vector):
    # Evaluate the system dynamics at the optimized state values
    derivatives = system_dynamics(DM([opt_CA, opt_T]))
    dCA_dt_values[i] = derivatives[0]
    dT_dt_values[i] = derivatives[1]

# Plot results
pio.templates.default = 'plotly_dark'
fig = make_subplots(cols=2, rows=1)
fig.add_scatter(x=time_vector, y=dCA_dt_values, mode='lines', name='dCA/dt', row=1, col=1)
fig.add_scatter(x=time_vector, y=dT_dt_values, mode='lines', name='dT/dt', row=1, col=2)
fig.show()