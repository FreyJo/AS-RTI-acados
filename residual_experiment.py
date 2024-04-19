# -*- coding: future_fstrings -*-
#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from typing import List
import numpy as np
import pickle, os

from setup_acados_integrator import setup_acados_integrator
from mpc_parameters import MpcPendulumParameters

from setup_acados_ocp_solver import (
    setup_acados_ocp_solver,
    augment_model_with_cost_state,
)

from models import setup_pendulum_model
from utils import plot_simulation_result, get_label_from_setting, get_results_filename, get_subdict, get_relevant_keys, KEY_TO_TEX
from simulate import simulate, simulate_with_residuals

dummy_model, dummy_model_params = setup_pendulum_model()
dummy_mpc_params = MpcPendulumParameters(xs=dummy_model_params.xs, us=dummy_model_params.us)
N_REF = int(dummy_mpc_params.T / dummy_mpc_params.dt)
DT_PLANT = dummy_mpc_params.dt


N_HORIZON = 20

OCP_NUM_STAGES = 2
SQP_SETTING = {
    "N": N_HORIZON,
    "sim_method_num_stages": OCP_NUM_STAGES,
    "algorithm": "SQP",
    "time_horizon": dummy_mpc_params.T,
    "nlp_solver_max_iter": 100,
}

REFERENCE_SETTING = SQP_SETTING.copy()
REFERENCE_SETTING["N"] = N_REF


RTI_SETTING = {
    "N": N_HORIZON,
    "sim_method_num_stages": OCP_NUM_STAGES,
    "algorithm": "RTI",
    "time_horizon": dummy_mpc_params.T,
    "nlp_solver_max_iter": 1,
}

AS_RTI_SETTING = RTI_SETTING.copy()
AS_RTI_SETTING["algorithm"] = "AS-RTI-D"
AS_RTI_SETTING["nlp_solver_max_iter"] = 1

AS_RTI_A_SETTING = AS_RTI_SETTING.copy()
AS_RTI_A_SETTING["algorithm"] = "AS-RTI-A"

AS_RTI_2_SETTING = AS_RTI_SETTING.copy()
AS_RTI_2_SETTING["nlp_solver_max_iter"] = 2

AS_RTI_B1_SETTING = AS_RTI_SETTING.copy()
AS_RTI_B1_SETTING["algorithm"] = "AS-RTI-B"
AS_RTI_B1_SETTING["nlp_solver_max_iter"] = 1

AS_RTI_B2_SETTING = AS_RTI_B1_SETTING.copy()
AS_RTI_B2_SETTING["nlp_solver_max_iter"] = 2

AS_RTI_B10_SETTING = AS_RTI_B1_SETTING.copy()
AS_RTI_B10_SETTING["nlp_solver_max_iter"] = 10

AS_RTI_D10_SETTING = AS_RTI_B10_SETTING.copy()
AS_RTI_D10_SETTING["algorithm"] = "AS-RTI-D"

AS_RTI_C1_SETTING = AS_RTI_B1_SETTING.copy()
AS_RTI_C1_SETTING["algorithm"] = "AS-RTI-C"

AS_RTI_C10_SETTING = AS_RTI_C1_SETTING.copy()
AS_RTI_C10_SETTING["nlp_solver_max_iter"] = 10


def run_residual_experiment(settings: List[dict]):
    Tsim = 2.0

    plant_model, plant_model_params = setup_pendulum_model()
    mpc_params = MpcPendulumParameters(xs=plant_model_params.xs, us=plant_model_params.us)

    # to compute LQR matrix P
    # linearized_model = setup_linearized_model(plant_model, plant_model_params, dummy_mpc_params)

    plant_model = augment_model_with_cost_state(plant_model, plant_model_params, mpc_params=mpc_params)
    plant_model.name += "_plant"

    Nsim = int(Tsim / DT_PLANT)
    if not (Tsim / DT_PLANT).is_integer():
        print("WARNING: Tsim / DT_PLANT should be an integer!")

    integrator = setup_acados_integrator(plant_model, DT_PLANT, mpc_params,
                    num_steps=mpc_params.sim_method_num_steps,
                    num_stages=mpc_params.sim_method_num_stages, integrator_type="IRK",
                    )

    labels_all = []
    nu = dummy_model_params.nu_original

    disturbance = np.zeros((Nsim, nu))
    disturbance[4, :] = 110.

    x0 = np.array([-0.05, 0.03, 0.0, 0.0])


    for setting in settings:
        label = get_label_from_setting(setting)

        model, model_params = setup_pendulum_model()

        mpc_params = MpcPendulumParameters(xs=model_params.xs, us=model_params.us)
        levenberg_marquardt=0.0

        mpc_params.N = setting['N']
        mpc_params.sim_method_num_stages = setting['sim_method_num_stages']

        ocp_solver = setup_acados_ocp_solver(model,
                model_params, mpc_params, levenberg_marquardt=levenberg_marquardt,
                hessian_approx='GAUSS_NEWTON',
                nlp_solver_max_iter=setting['nlp_solver_max_iter'],
                algorithm = setting["algorithm"],
                integrator_type="IRK",
                rti_log_residuals=1
                )

        if setting["algorithm"] in ["AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
            if setting['algorithm'] == 'AS-RTI-A':
                ocp_solver.options_set('as_rti_level', 0)
            elif setting['algorithm'] == 'AS-RTI-B':
                ocp_solver.options_set('as_rti_level', 1)
            elif setting['algorithm'] == 'AS-RTI-C':
                ocp_solver.options_set('as_rti_level', 2)
            elif setting['algorithm'] == 'AS-RTI-D':
                ocp_solver.options_set('as_rti_level', 3)
        print(f"{setting=}, {mpc_params.T}")

        print(f"\n\nRunning CLOSED loop simulation with {label}\n\n")

        results = simulate_with_residuals(ocp_solver, integrator, model_params, x0, Nsim,
                        controller_setting=setting,
                        disturbance=disturbance)

        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, True, "residual")
        results['label'] = label
        results['mpc_params'] = mpc_params
        labels_all.append(label)
        pickle.dump(results, open(results_filename, "wb"))
        print(f"saved result as {results_filename}")
        ocp_solver = None


def get_latex_label_from_setting(setting: dict):
    label = ''
    for key, value in setting.items():
        if key == 'use_rti':
            if value == False:
                label += 'SQP '
            else:
                label += 'RTI '
        else:
            label += f"{KEY_TO_TEX[key]} {value} "
    return label

def plot_trajectories(settings, labels=None, ncol_legend=1, title=None, bbox_to_anchor=None, fig_filename=None):
    X_all = []
    U_all = []
    labels_all = []

    relevant_keys, constant_keys = get_relevant_keys(settings)
    common_description = get_latex_label_from_setting(get_subdict(settings[0], constant_keys))

    # load
    for i, setting in enumerate(settings):
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, True, "residual")

        descriptive_setting = get_subdict(setting, relevant_keys)

        if labels is not None:
            latex_label = labels[i]
        else:
            latex_label = get_latex_label_from_setting(descriptive_setting)
        # check if file exists
        if not os.path.exists(results_filename):
            print(f"File {results_filename} corresponding to {label} does not exist.")
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)

        X_all.append(results['x_traj'])
        U_all.append(results['u_traj'])
        # labels_all.append(results['label'])
        labels_all.append(latex_label)


    x_lables_list = ["$p$ [m]", r"$\theta$ [rad/s]", "$s$ [m/s]", r"$\dot{\theta}$", r"cost state"]
    u_lables_list = ["$u$ [N]"]

    if title is None:
        title = common_description

    plot_simulation_result(
        DT_PLANT,
        X_all,
        U_all,
        dummy_mpc_params.umin,
        dummy_mpc_params.umax,
        x_lables_list,
        u_lables_list,
        labels_all,
        idxpx=[0, 1],
        title=title,
        linestyle_list=['--', ':', '--', ':', '--', '-.', '-.', ':'],
        single_column=True,
        xlabel='$t$ [s]',
        idx_xlogy= [4],
        ncol_legend = ncol_legend,
        # color_list=['C0', 'C0', 'C1', 'C1']
        fig_filename=fig_filename,
        bbox_to_anchor=bbox_to_anchor,
    )

def plot_residuals(settings, labels, i_start=0, i_end=-1, fig_filename=None):
    from matplotlib import pyplot as plt
    from acados_template import latexify_plot
    latexify_plot()
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'D', 'P', '*', 'X']
    marker_sizes = [4, 3, 4, 4, 4, 8, 8, 5, 5]
    for i_setting, (setting, plot_label, marker, marker_size) in enumerate(zip(settings, labels, markers, marker_sizes)):
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, True, "residual")

        # check if file exists
        if not os.path.exists(results_filename):
            print(f"File {results_filename} corresponding to {label} does not exist.")
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)

        res_eq_list = results['res_eq_list'][i_start:i_end]
        res_stat_list = results['res_stat_list'][i_start:i_end]
        max_length = max([len(r) for r in res_eq_list])
        n_time_steps = len(res_eq_list)
        x_vals = np.linspace(i_start+1/max_length, i_start+n_time_steps, n_time_steps*max_length)
        res_eq_vals = np.nan * np.ones(x_vals.shape)
        res_stat_vals = np.nan * np.ones(x_vals.shape)

        for i, (res_eq, res_stat) in enumerate(zip(res_eq_list, res_stat_list)):
            iter = len(res_eq)
            res_eq_vals[i*max_length:i*max_length+iter] = res_eq
            res_stat_vals[i*max_length:i*max_length+iter] = res_stat
        print(f"\n{plot_label}: {res_eq_list[2:4]=}, {max_length=}, {x_vals=}")

        color = f"C{i_setting}"
        for plot_idx, res in zip([0, 1], [res_eq_vals, res_stat_vals]):
            axs[plot_idx].plot(x_vals, res, color=color, alpha=0.5,
                        # label=plot_label,
                        marker=marker, linestyle='', markersize=marker_size)
            axs[plot_idx].plot(x_vals[max_length-1:n_time_steps*max_length:max_length],
                        res[max_length-1:n_time_steps*max_length:max_length],
                        color=color,
                        alpha=1.0,
                        label=plot_label,
                        marker=marker, linestyle='', markersize=2*marker_size)

    axs[1].set_xlabel('time step')
    axs[0].set_ylabel(r'$|| g || $')
    axs[1].set_ylabel(r'$ || \nabla_x \mathcal{L} ||_\infty$')
    l_handles, l_labels = axs[0].get_legend_handles_labels()

    from matplotlib.lines import Line2D

    custom_handles = [
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    alpha=0.5,
                    markersize=8,
                    markeredgewidth=2,
                    fillstyle='full',
                    color='C0',
                    lw=0,
                    label='inner iterations',
                ),
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    alpha=1.0,
                    markersize=8,
                    markeredgewidth=2,
                    fillstyle='full',
                    color='C0',
                    lw=0,
                    label='feedback',
                ),
    ]
    legend = axs[1].legend(handles=l_handles + custom_handles, loc="lower center",
                           bbox_to_anchor=(0.66, -0.03),
                           columnspacing=0.3,
                           handletextpad=0.2,
                        #    bbox_to_anchor=(0.45, -1.),
                           ncol=3)

    axs[0].set_xlim([i_start, i_start+n_time_steps])
    from matplotlib.ticker import MaxNLocator
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid()
    axs[0].grid()
    plt.tight_layout()

    if fig_filename is not None:
        fig_filename = os.path.join("figures", fig_filename)
        plt.savefig(fig_filename, bbox_inches='tight')
        print(f"saved figure as {fig_filename}")
    plt.show()



if __name__ == "__main__":

    SETTING_LABEL_PAIRS = [
        (AS_RTI_D10_SETTING, "AS-RTI-D-10"),
        (AS_RTI_C10_SETTING, 'AS-RTI-C-10'),
        (AS_RTI_B10_SETTING, "AS-RTI-B-10"),
        (AS_RTI_A_SETTING, "AS-RTI-A"),
        (RTI_SETTING, "RTI"),
        ]
    SETTINGS = [setting for setting, label in SETTING_LABEL_PAIRS]
    LABELS = [label for setting, label in SETTING_LABEL_PAIRS]

    run_residual_experiment(SETTINGS)

    # plot_trajectories(SETTINGS,
    #                 labels = LABELS,
    #                 ncol_legend=3,
    #                 title="",
    #                 bbox_to_anchor=(0.5, -0.85),
    #             )

    plot_residuals(SETTINGS, LABELS, 4, 10, fig_filename="as_rti_inner_residuals.pdf")