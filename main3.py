from cstr_model import CSTRParameters, setup_cstr_model, setup_linearized_model
from setup_acados_ocp_solver import (
    MpcCSTRParameters,
    setup_acados_ocp_solver,
    AcadosOcpSolver,
)
from setup_acados_integrator import setup_acados_integrator, AcadosSimSolver
import numpy as np
from cstr_utils import plot_cstr
from typing import Optional
import tensorflow as tf
from joblib import load
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib.pyplot as plt

# Load the trained model and scalers outside the function
loaded_model = tf.keras.models.load_model('cstr_model.keras')
scaler_X = load('scaler_X.joblib')
scaler_y = load('scaler_y.joblib')

def generate_predicted_references(current_state, current_control):
    combined_input = np.concatenate((current_state, current_control))

    combined_input_scaled = scaler_X.transform(combined_input.reshape(1, -1))

    # Predict
    predicted_y_scaled = loaded_model.predict(combined_input_scaled)

    predicted_references = scaler_y.inverse_transform(predicted_y_scaled)
    return predicted_references

def simulate(
    controller: Optional[AcadosOcpSolver],
    plant: AcadosSimSolver,
    x0: np.ndarray,
    Nsim: int,
    X_ref: np.ndarray,
    U_ref: np.ndarray,
):

    nx = X_ref.shape[1]
    nu = U_ref.shape[1]

    X = np.ndarray((Nsim + 1, nx))
    U = np.ndarray((Nsim, nu))
    timings_solver = np.zeros((Nsim))
    timings_integrator = np.zeros((Nsim))

    # closed loop
    xcurrent = x0
    X[0, :] = xcurrent

    for i in range(Nsim):

        if controller is None:
            U[i, :] = np.array([300 , 0.1]) 
        else:
            # set initial state
            controller.set(0, "lbx", xcurrent)
            controller.set(0, "ubx", xcurrent)

            yref = np.concatenate((X_ref[i, :], U_ref[i, :]))
            for stage in range(controller.acados_ocp.dims.N):
                controller.set(stage, "yref", yref)
            controller.set(controller.acados_ocp.dims.N, "yref", X_ref[i, :])

            # solve ocp
            status = controller.solve()

            if status != 0:
                controller.print_statistics()
                raise Exception(
                    f"acados controller returned status {status} in simulation step {i}. Exiting."
                )

            U[i, :] = controller.get(0, "u")
            timings_solver[i] = controller.get_stats("time_tot")

        # simulate system
        plant.set("x", xcurrent)
        plant.set("u", U[i, :])

        if plant.acados_sim.solver_options.integrator_type == "IRK":
            plant.set("xdot", np.zeros((nx,)))

        status = plant.solve()
        if status != 0:
            raise Exception(
                f"acados integrator returned status {status} in simulation step {i}. Exiting."
            )

        timings_integrator[i] = plant.get("time_tot")
        # update state
        xcurrent = plant.get("x")
        X[i + 1, :] = xcurrent

    return X, U, timings_solver, timings_integrator

def simulate_nn(controller: Optional[AcadosOcpSolver], plant: AcadosSimSolver, x0: np.ndarray, Nsim: int):
    nx = 3  
    nu = 2  

    X = np.ndarray((Nsim + 1, nx))
    U = np.ndarray((Nsim, nu))
    timings_solver = np.zeros((Nsim))
    timings_integrator = np.zeros((Nsim))

    xcurrent = x0
    X[0, :] = xcurrent

    for i in range(Nsim):
      
        current_control = np.array([300, 0.1])  # Update this as per your simulation's requirement
        predicted_refs = generate_predicted_references(xcurrent, current_control)

        # Extract references for the current step
        X_ref_pred = predicted_refs[0, :nx]  
        U_ref_pred = predicted_refs[0, nx:]  

        if controller is not None:
            controller.set(0, "lbx", xcurrent)
            controller.set(0, "ubx", xcurrent)

            yref = np.concatenate((X_ref_pred, U_ref_pred))
            for stage in range(controller.acados_ocp.dims.N):
                controller.set(stage, "yref", yref)
            controller.set(controller.acados_ocp.dims.N, "yref", X_ref_pred)

            status = controller.solve()
            if status != 0:
                controller.print_statistics()
                raise Exception(f"acados controller returned status {status} in simulation step {i}.")

            U[i, :] = controller.get(0, "u")
            timings_solver[i] = controller.get_stats("time_tot")

        else:
            U[i, :] = U_ref_pred

        plant.set("x", xcurrent)
        plant.set("u", U[i, :])

        if plant.acados_sim.solver_options.integrator_type == "IRK":
            plant.set("xdot", np.zeros((nx,)))

        status = plant.solve()
        if status != 0:
            raise Exception(f"acados integrator returned status {status} in simulation step {i}.")

        timings_integrator[i] = plant.get("time_tot")
        xcurrent = plant.get("x")
        X[i + 1, :] = xcurrent

    return X, U, timings_solver, timings_integrator

def plot_cstr(
    dt,
    X_list,
    U_list,
    X_ref,
    U_ref,
    u_min,
    u_max,
    labels_list,
    fig_filename=None,
    x_min=None,
    x_max=None,
):
    """
    Params:

    """

    nx = X_list[0].shape[1]
    nu = U_list[0].shape[1]

    Nsim = U_list[0].shape[0]

    ts = dt * np.arange(0, Nsim + 1)

    states_lables = ["$c$ [kmol/m$^3$]", "$T$ [K]", "$h$ [m]"]
    controls_lables = ["$T_c$ [K]", "$F$ [m$^3$/min]"]

    fig, axes = plt.subplots(ncols=2, nrows=nx)

    for i in range(nx):
        for X, label in zip(X_list, labels_list):
            axes[i, 0].plot(ts, X[:, i], label=label, alpha=1)

        axes[i, 0].step(
            ts,
            X_ref[:, i],
            alpha=1,
            where="post",
            label="reference",
            linestyle="--",
            color="k",
        )
        axes[i, 0].set_ylabel(states_lables[i])
      
        axes[i, 0].set_xlim(ts[0], ts[-1])

     

    for i in range(nu):
        for U, label in zip(U_list, labels_list):
            axes[i, 1].step(ts, np.append([U[0, i]], U[:, i]), label=label, alpha=1)
        axes[i, 1].step(
            ts,
            np.append([U_ref[0, i]], U_ref[:, i]),
            alpha=1,
            label="reference",
            linestyle="--",
            color="k",
        )
        axes[i, 1].set_ylabel(controls_lables[i])


    axes[1, 1].legend(bbox_to_anchor=(0.5, -1.25), loc="lower center")
    axes[-1, 0].set_xlabel("$t$ [min]")
    axes[1, 1].set_xlabel("$t$ [min]")

    fig.delaxes(axes[-1, 1])

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, hspace=0.3, wspace=0.4
    )
    if fig_filename is not None:
        # TODO: legend covers x label :O
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    plt.show()




def main():

    Tsim = 25
    dt_plant = 0.25  # [min]

    cstr_params = CSTRParameters()
    mpc_params = MpcCSTRParameters(xs=cstr_params.xs, us=cstr_params.us)
    model = setup_cstr_model(cstr_params)
    linearized_model = setup_linearized_model(model, cstr_params, mpc_params)
    plant_model = setup_cstr_model(cstr_params)

    Nsim = int(Tsim / dt_plant)
    if not (Tsim / dt_plant).is_integer():
        print("WARNING: Tsim / dt_plant should be an integer!")

    integrator = setup_acados_integrator(plant_model, dt_plant, cstr_param=cstr_params)

    # steady-state
    xs = np.array([[0.878, 324.5, 0.659]]).T
    us = np.array([[300, 0.1]]).T

    # constant ref
    X_ref = np.tile(xs, Nsim + 1).T
    U_ref = np.tile(us, Nsim).T

  
    x0 = np.array([0.05, 0.75, 0.5]) * xs.ravel()

    X_all = []
    U_all = []
    labels_all = []
    timings_solver_all = []

    label = "DNN-NMPC"
    print(f"\n\nRunning simulation with {label}\n\n")
    ocp_solver = setup_acados_ocp_solver(
        model, mpc_params, cstr_params=cstr_params, use_rti=True
    )

    X, U, timings_solver, _ = simulate_nn(
        ocp_solver, integrator, x0, Nsim
    )
    X_all.append(X)
    U_all.append(U)
    timings_solver_all.append(timings_solver)
    labels_all.append(label)
    ocp_solver = None
    
    label = "LMPC"
    print(f"\n\nRunning simulation with {label}\n\n")
    mpc_params.linear_mpc = True
    ocp_solver = setup_acados_ocp_solver(
        linearized_model, mpc_params, cstr_params=cstr_params, use_rti=True
    )
    mpc_params.linear_mpc = False

    X, U, timings_solver, _ = simulate(
        ocp_solver, integrator, x0, Nsim, X_ref=X_ref, U_ref=U_ref
    )
    X_all.append(X)
    U_all.append(U)
    timings_solver_all.append(timings_solver)
    labels_all.append(label)
    ocp_solver = None

    plot_cstr(
        dt_plant,
        X_all,
        U_all,
        X_ref,
        U_ref,
        mpc_params.umin,
        mpc_params.umax,
        labels_all,
        fig_filename='cstr_acados.pdf',
    )  # , fig_filename='cstr_acados.pdf')


if __name__ == "__main__":
    main()
