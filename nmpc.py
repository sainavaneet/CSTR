import casadi as ca
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
# System parameters
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

# NMPC setup
# NMPC setup
N = 10  # Number of control intervals in the prediction horizon
total_time = 10.0  # Total time horizon (rename this variable)
h = total_time/N  # Length of each control interval

# Define the system dynamics
def system_dynamics(CA, T, Tc):
    K = K0 * ca.exp(-Ac/T)
    dCA_dt = q/V*(CAin-CA) - K*CA
    dT_dt = q/V*(Tin-T) + (HR*K*CA)/(p*C) - UA*(T-Tc)/(p*V*C)
    return ca.vertcat(dCA_dt, dT_dt)

# States and control variables
CA = ca.MX.sym('CA')
T = ca.MX.sym('T')
Tc = ca.MX.sym('Tc')  # Control variable: Cooling temperature

# Objective and constraints
obj = 0  # Objective function
g = []  # Constraints vector

# Initial state
CA0 = 0.5
T0 = 350.
x0 = ca.vertcat(CA0, T0)

# Start with an initial guess for control inputs
U = ca.MX.sym('U', N)

# Formulate the NMPC optimization problem
X = x0
for i in range(N):
    # Control input for this interval
    Tc = U[i]
    

    # Integrate over this interval
    k1 = system_dynamics(X[0], X[1], Tc)
    k2 = system_dynamics(X[0] + h/2*k1[0], X[1] + h/2*k1[1], Tc)
    k3 = system_dynamics(X[0] + h/2*k2[0], X[1] + h/2*k2[1], Tc)
    k4 = system_dynamics(X[0] + h*k3[0], X[1] + h*k3[1], Tc)
    X_next = X + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    

    # Update the objective function (simple example: minimize T deviation)
    obj += (X_next[1] - Tin)**2

    # Add constraints (e.g., temperature bounds)
    g.append(X_next[1] - 300)  # Lower bound on T
    g.append(400 - X_next[1])  # Upper bound on T

    # Update the state for the next iteration
    X = X_next



# Setup the optimizer
opts = {"ipopt.print_level": 0, "print_time": 0}
nlp = {'x': U, 'f': obj, 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Bounds on U and g
lbu = [290] * N  # Lower bound on Tc
ubu = [310] * N  # Upper bound on Tc
lbg = [0] * 2*N  # Lower bound on g
ubg = [ca.inf] * 2*N  # Upper bound on g

# Solve the NMPC optimization problem
sol = solver(lbx=lbu, ubx=ubu, lbg=lbg, ubg=ubg)
u_opt = sol['x']

print("Optimal control sequence: ", u_opt)


time = np.linspace(0, total_time, N)

# Assuming u_opt is the optimal control sequence you obtained
plt.figure(figsize=(10, 5))
plt.plot(time, u_opt, marker='o')
plt.title('Optimal Cooling Temperature Control Sequence')
plt.xlabel('Time')
plt.ylabel('Cooling Temperature (Tc)')
plt.grid(True)
plt.show()