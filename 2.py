import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

# Given parameters
q = 100  # L/min
cA_i = 1  # mol/L
T_i = 350  # K
V = 100  # L
rho = 1000 # g/L
C = 0.239 # J/(g K)
Hr = -5e4  # J/(mol)
E_over_R = 8750  # K
k0 = 7.2e10  # 1/min
UA = 5e4  # J/(min K)
Tc0 = 300  # K

cA0 = 0.5  # mol/L
T0 = 350  # K

# Reaction rate function
def reaction_rate(cA, T):
    k = k0 * np.exp(-E_over_R / T)
    return k * cA

# Differential equations
def cstr_model(y, t):
    cA, T = y
    Tc = Tc0  # Assuming constant coolant temperature
    
    # Material balance for A
    dcAdt = q/V * (cA_i - cA) - reaction_rate(cA, T)
    
    # Energy balance
    dTdt = q/V * (T_i - T) + (-Hr / (rho * C)) * reaction_rate(cA, T) + UA/V/(rho*C) * (Tc - T)
    
    return [dcAdt, dTdt]

# Time grid (min)
t = np.linspace(0, 10, 100)  # for 10 minutes

# Initial conditions
y0 = [cA0, T0]

# Solve the ODE
solution = odeint(cstr_model, y0, t)

# Extracting results
cA = solution[:, 0]
T = solution[:, 1]

# Plotting
fig = go.Figure()

# Concentration Plot
fig.add_trace(go.Scatter(x=t, y=cA, mode='lines', name='Concentration (mol/L)'))

# Temperature Plot
fig.add_trace(go.Scatter(x=t, y=T, mode='lines', name='Temperature (K)', yaxis='y2'))

# Layout
fig.update_layout(
    title="CSTR Concentration and Temperature over Time",
    xaxis_title="Time (min)",
    yaxis_title="Concentration (mol/L)",
    yaxis2=dict(
        title="Temperature (K)",
        overlaying='y',
        side='right'
    )
)

fig.show()

