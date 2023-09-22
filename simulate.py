from typing import Optional
import numpy as np
from acados_template import (
    AcadosOcpSolver, AcadosSimSolver, AcadosMultiphaseOcp, AcadosOcp
)
from models import ModelParameters


def initialize_controller(controller: AcadosOcpSolver, model_params: ModelParameters, x0: np.ndarray):
    time_steps = controller.acados_ocp.solver_options.time_steps
    if isinstance(controller.acados_ocp, AcadosOcp):
        nx_ocp = controller.acados_ocp.dims.nx
    elif isinstance(controller.acados_ocp, AcadosMultiphaseOcp):
        nx_ocp = controller.acados_ocp.phases_dims[0].nx

    # append zeros to x0 for
    nx0 = x0.size
    x0_ocp = np.hstack((x0, np.zeros(nx_ocp-nx0)))
    cost_guess = 0.0

    t = 0.0
    T = sum(time_steps)
    controller.set(0, 'x', x0_ocp)
    for i, dt in enumerate(time_steps):
        u_guess = model_params.us
        # linspace in time between x0 and xs
        x_guess = t/T*model_params.xs + (T-t)/T * x0_ocp

        if model_params.cost_state_idx is not None:
            cost_guess += dt * model_params.cost_state_dyn_fun(x_guess, u_guess, model_params.xs[-1], model_params.us).full().item()
            x_guess[model_params.cost_state_idx] = cost_guess
            # print(f"{x_guess=}")

        x_init = x_guess

        controller.set(i+1, 'x', x_init)
        if i<controller.N:
            if isinstance(controller.acados_ocp, AcadosOcp):
                controller.set(i, 'u', u_guess)
            elif isinstance(controller.acados_ocp, AcadosMultiphaseOcp):
                controller.set(i, 'u', 0*controller.get(i, 'u'))
            t += dt

    # last stage
    controller.set(controller.N, 'x', x_init)

    return


def simulate(
    controller: Optional[AcadosOcpSolver],
    plant: AcadosSimSolver,
    model_params: ModelParameters,
    x0: np.ndarray,
    Nsim: int,
    controller_setting: dict,
    n_runs = 1,
    disturbance: Optional[np.ndarray] = None,
):

    nx = plant.acados_sim.dims.nx
    nu = plant.acados_sim.dims.nu

    if isinstance(controller.acados_ocp, AcadosOcp):
        nx_ocp = controller.acados_ocp.dims.nx
    elif isinstance(controller.acados_ocp, AcadosMultiphaseOcp):
        nx_ocp = controller.acados_ocp.phases_dims[0].nx

    X_sim = np.nan * np.ones((Nsim + 1, nx))
    U_sim = np.nan * np.ones((Nsim, nu))

    X_ref = np.tile(model_params.xs, (Nsim+1, 1))
    U_ref = np.tile(model_params.us, (Nsim, 1))
    timings_solver = np.zeros((Nsim))
    timings_integrator = np.zeros((Nsim))
    timings_preparation = np.zeros((Nsim))
    timings_feedback = np.zeros((Nsim))
    nlp_iter = np.zeros((Nsim))
    res_eq = np.zeros((Nsim))
    res_stat = np.zeros((Nsim))
    for irun in range(n_runs):
        if controller is not None:
            controller.reset()
            # initialize_controller(controller, model_params, x0)
            initialize_controller(controller, model_params, model_params.xs)

        # closed loop
        xcurrent = np.concatenate((x0.T, np.zeros((nx-model_params.nx_original))))
        X_sim[0, :] = xcurrent
        time_prep = 0.0
        if controller_setting["algorithm"] in ["RTI", "AS-RTI-py"]:
            # first preparation phase
            controller.options_set("rti_phase", 1)
            status = controller.solve()
            time_prep = controller.get_stats("time_tot")
            if controller_setting["algorithm"] == "AS-RTI-py":
                controller.store_iterate('as_rti_iter.json', overwrite=True)
        elif controller_setting["algorithm"] in ["AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
            # first preparation phase
            controller.options_set("rti_phase", 3)
            status = controller.solve()
            time_prep = controller.get_stats("time_tot")


        for i in range(Nsim):

            x0_bar = xcurrent[:nx_ocp]

            controller.set(0, "lbx", x0_bar)
            controller.set(0, "ubx", x0_bar)

            # feedback phase
            if controller_setting["algorithm"] == "SQP":
                status = controller.solve()
                timing = controller.get_stats("time_tot")
                time_feedback = timing

            elif controller_setting["algorithm"] in ["RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
                # feedback
                controller.options_set("rti_phase", 2)
                status = controller.solve()
                time_feedback = controller.get_stats("time_tot")

                timing = time_feedback + time_prep

            U_sim[i, :] = controller.get(0, "u")
            if np.isnan(U_sim[i, :]).any():
                print("u is nan")
                controller.dump_last_qp_to_json("failing_qp.json", overwrite=True)
                status = 4
                controller.reset()

            if status not in [0, 2]:
                controller.print_statistics()
                msg = f"acados controller returned status {status} in simulation step {i}."
                print("warning: " + msg + "\n\n\nEXITING with unfinished simulation.\n\n\n")
                controller.reset()

                controller.dump_last_qp_to_json("failing_qp.json", overwrite=True)
                controller.store_iterate("failing_iterate.json", overwrite=True)

                return dict(x_traj=X_sim, u_traj=U_sim, timings=timings_solver, timings_preparation=timings_preparation, timings_feedback=timings_feedback, x_ref=X_ref, u_ref=U_ref, status=status, nlp_iter=nlp_iter, res_eq=res_eq, res_stat=res_stat)

            if disturbance is not None and disturbance[i, :] != 0:
                U_sim[i, :] = disturbance[i, :]
                # print(f"disturbance applied in step {i}.")
            if irun == 0:
                timings_solver[i] = timing
                timings_feedback[i] = time_feedback
                timings_preparation[i] = time_prep
            else:
                timings_solver[i] = min(timing, timings_solver[i])
                timings_feedback[i] = min(time_feedback, timings_feedback[i])
                timings_preparation[i] = min(time_prep, timings_preparation[i])

            nlp_iter[i] = controller.get_stats("sqp_iter")

            # simulate system
            plant.set("x", xcurrent)
            plant.set("u", U_sim[i, :])

            if plant.acados_sim.solver_options.integrator_type == "IRK":
                plant.set("xdot", np.zeros((nx,)))

            status = plant.solve()
            if status != 0:
                raise Exception(
                    f"acados integrator returned status {status} in simulation step {i}. Exiting."
                )

            timings_integrator[i] = plant.get("time_tot")

            # log additional quantities for evaluation
            if controller is not None and irun == 0:
                residuals = controller.get_residuals()
                res_stat[i] = residuals[0]
                res_eq[i] = residuals[1]


            # preparation step
            if controller_setting["algorithm"] in ["RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
                if controller_setting["algorithm"] == "RTI":
                    controller.options_set("rti_phase", 1)
                elif controller_setting["algorithm"].startswith("AS-RTI"):
                    controller.options_set("rti_phase", 3)
                status = controller.solve()
                time_prep = controller.get_stats("time_tot")
                if status not in [0, 2]:
                    controller.print_statistics()
                    msg = f"acados controller returned status {status} in simulation step {i}."
                    print("warning: " + msg + "\n\n\nEXITING with unfinished simulation.\n\n\n")
                    controller.reset()

                    controller.dump_last_qp_to_json("failing_qp.json", overwrite=True)
                    controller.store_iterate("failing_iterate.json", overwrite=True)

                    return dict(x_traj=X_sim, u_traj=U_sim, timings=timings_solver, timings_preparation=timings_preparation, timings_feedback=timings_feedback, x_ref=X_ref, u_ref=U_ref, status=status, nlp_iter=nlp_iter, res_eq=res_eq, res_stat=res_stat)

            # update state
            xcurrent = plant.get("x")
            X_sim[i + 1, :] = xcurrent

    if np.isnan(res_eq).any():
        print("warning: residuals are nan")
        breakpoint()

    return dict(x_traj=X_sim, u_traj=U_sim, timings=timings_solver, timings_preparation=timings_preparation, timings_feedback=timings_feedback, x_ref=X_ref, u_ref=U_ref, status=status, nlp_iter=nlp_iter, res_eq=res_eq, res_stat=res_stat)





def simulate_with_residuals(
    controller: Optional[AcadosOcpSolver],
    plant: AcadosSimSolver,
    model_params: ModelParameters,
    x0: np.ndarray,
    Nsim: int,
    controller_setting: dict,
    disturbance: Optional[np.ndarray] = None,
):

    collect_residuals = True
    nx = plant.acados_sim.dims.nx
    nu = plant.acados_sim.dims.nu

    if isinstance(controller.acados_ocp, AcadosOcp):
        nx_ocp = controller.acados_ocp.dims.nx
    elif isinstance(controller.acados_ocp, AcadosMultiphaseOcp):
        nx_ocp = controller.acados_ocp.phases_dims[0].nx

    X_sim = np.nan * np.ones((Nsim + 1, nx))
    U_sim = np.nan * np.ones((Nsim, nu))

    controller.reset()
    initialize_controller(controller, model_params, model_params.xs)

    res_stat_list = []
    res_eq_list = []

    # closed loop
    xcurrent = np.concatenate((x0.T, np.zeros((nx-model_params.nx_original))))
    X_sim[0, :] = xcurrent
    if controller_setting["algorithm"] == "RTI":
        # first preparation phase
        controller.options_set("rti_phase", 1)
        status = controller.solve()
    elif controller_setting["algorithm"] in ["AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
        # first preparation phase
        controller.options_set("rti_phase", 3)
        status = controller.solve()

    for i in range(Nsim):

        # call controller
        x0_bar = xcurrent[:nx_ocp]

        controller.set(0, "lbx", x0_bar)
        controller.set(0, "ubx", x0_bar)

        # feedback phase
        if controller_setting["algorithm"] == "SQP":
            status = controller.solve()

        elif controller_setting["algorithm"] in ["RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
            # feedback
            controller.options_set("rti_phase", 2)
            status = controller.solve()

        if collect_residuals:
            res_eq_list.append(controller.get_stats("res_eq_all"))
            res_stat_list.append(controller.get_stats("res_stat_all"))

        U_sim[i, :] = controller.get(0, "u")
        if np.isnan(U_sim[i, :]).any():
            print("u is nan")
            controller.dump_last_qp_to_json("failing_qp.json", overwrite=True)
            status = 4
            controller.reset()

        # controller.print_statistics()
        if status not in [0, 2]:
            raise Exception(f"controller returned status {status} in feedback phase")

        if disturbance is not None and disturbance[i, :] != 0:
            U_sim[i, :] = disturbance[i, :]
            print(f"disturbance applied in step {i}.")

        # simulate system
        plant.set("x", xcurrent)
        plant.set("u", U_sim[i, :])

        if plant.acados_sim.solver_options.integrator_type == "IRK":
            plant.set("xdot", np.zeros((nx,)))

        status = plant.solve()
        if status != 0:
            raise Exception(
                f"acados integrator returned status {status} in simulation step {i}. Exiting."
            )

        # log additional quantities for evaluation
        controller.get_stats("statistics")

        # preparation step
        if controller_setting["algorithm"] in ["RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
            if controller_setting["algorithm"] == "RTI":
                controller.options_set("rti_phase", 1)
            elif controller_setting["algorithm"].startswith("AS-RTI"):
                controller.options_set("rti_phase", 3)
            status = controller.solve()
            if status not in [0, 2]:
                raise Exception(f"controller returned status {status} in preparation phase step {i}")
        else:
            raise NotImplementedError("only RTI and AS-RTI is implemented")

        # update state
        xcurrent = plant.get("x")
        X_sim[i + 1, :] = xcurrent

    return dict(x_traj=X_sim, u_traj=U_sim, res_eq_list=res_eq_list, res_stat_list=res_stat_list)


