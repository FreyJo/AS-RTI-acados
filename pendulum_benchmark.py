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

from typing import List, Optional
import numpy as np
import pickle, os
import itertools

from setup_acados_integrator import setup_acados_integrator
from mpc_parameters import MpcPendulumParameters

from setup_acados_ocp_solver import (
    setup_acados_ocp_solver,
    augment_model_with_cost_state,
)

from models import setup_pendulum_model, setup_linearized_model
from utils import plot_simulation_result, get_label_from_setting, get_results_filename, get_subdict, get_relevant_keys, KEY_TO_TEX, TEX_FOLDER, plot_simple_pareto
from simulate import simulate

dummy_model, dummy_model_params = setup_pendulum_model()
dummy_mpc_params = MpcPendulumParameters(xs=dummy_model_params.xs, us=dummy_model_params.us)
N_REF = int(dummy_mpc_params.T / dummy_mpc_params.dt)
DT_PLANT = dummy_mpc_params.dt


N_RUNS = 20
N_HORIZON = 20
N_SCENARIOS = 20

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

AS_RTI_C3_SETTING = AS_RTI_C1_SETTING.copy()
AS_RTI_C3_SETTING["nlp_solver_max_iter"] = 3

AS_RTI_C2_SETTING = AS_RTI_C1_SETTING.copy()
AS_RTI_C2_SETTING["nlp_solver_max_iter"] = 2

AS_RTI_C10_SETTING = AS_RTI_C1_SETTING.copy()
AS_RTI_C10_SETTING["nlp_solver_max_iter"] = 10

AS_RTI_C5_SETTING = AS_RTI_C1_SETTING.copy()
AS_RTI_C5_SETTING["nlp_solver_max_iter"] = 5

AS_RTI_C8_SETTING = AS_RTI_C1_SETTING.copy()
AS_RTI_C8_SETTING["nlp_solver_max_iter"] = 8

AS_RTI_GRID_SETTINGS = [AS_RTI_A_SETTING]
variants = ["AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]
for iter, algorithm in itertools.product(range(1, 11), variants):
    setting = AS_RTI_SETTING.copy()
    setting["nlp_solver_max_iter"] = iter
    setting["algorithm"] = algorithm
    AS_RTI_GRID_SETTINGS.append(setting)


RTI_2_SETTING = {
    "N": N_HORIZON,
    "sim_method_num_stages": OCP_NUM_STAGES,
    "algorithm": "SQP",
    "nlp_solver_max_iter": 2,
    "time_horizon": dummy_mpc_params.T,
}


RTI_3_SETTING = {
    "N": N_HORIZON,
    "sim_method_num_stages": OCP_NUM_STAGES,
    "algorithm": "SQP",
    "nlp_solver_max_iter": 3,
    "time_horizon": dummy_mpc_params.T,
}


def run_pendulum_benchmark_closed_loop(settings: List[dict]):
    Tsim = 4.0

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
                    num_stages=mpc_params.sim_method_num_stages, integrator_type="IRK")

    labels_all = []
    nu = dummy_model_params.nu_original

    N_1sec = int(1.0 / DT_PLANT)
    disturbance_base = np.zeros((Nsim, nu))
    dist_instances = [0, 2*N_1sec]

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    disturbance_list = []
    np.random.seed(0)
    if N_SCENARIOS > 1:
        x0_list = [np.array([p0, 0, 0, 0]) for p0 in np.linspace(-1., 1., N_SCENARIOS)]
        for k in range(N_SCENARIOS):
            disturbance = disturbance_base.copy()
            for i in dist_instances:
                disturbance[i, 0] = 2.5 * dummy_mpc_params.umax * (2*np.random.rand() - 1)
            disturbance_list.append(disturbance)
    else:
        x0_list = [x0]


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
                algorithm = setting["algorithm"]
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

        for i_scenario, (x0, disturbance) in enumerate(zip(x0_list, disturbance_list)):
            results = simulate(ocp_solver, integrator, model_params, x0, Nsim, n_runs=N_RUNS,
                            controller_setting=setting,
                            disturbance=disturbance)
            results = add_total_cost_to_results(results)

            results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, True, i_scenario)
            results['label'] = label
            results['mpc_params'] = mpc_params
            labels_all.append(label)
            pickle.dump(results, open(results_filename, "wb"))
            print(f"saved result as {results_filename}")
        ocp_solver = None

    print(f"ran all experiments with {len(labels_all)} different settings")

def add_total_cost_to_results(results):
    x_end = results['x_traj'][-1]
    terminal_cost_state = x_end[-1]
    x_og_vec = np.expand_dims(x_end[:dummy_model_params.nx_original], axis=1)
    terminal_cost_term = (x_og_vec.T @ dummy_mpc_params.P @ x_og_vec)[0][0]
    # intermediate terminal cost
    # i_mid = int(results['x_traj'].shape[0]/2)
    # x_mid = results['x_traj'][i_mid]
    # x_og_vec = np.expand_dims(x_mid[:dummy_model_params.nx_original], axis=1)
    # intermediate_terminal_cost_term = (x_og_vec.T @ dummy_mpc_params.P @ x_og_vec)[0][0]
    results['cost_total'] = terminal_cost_state + terminal_cost_term
    return results

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

def plot_trajectories(settings, labels=None, ncol_legend=1, title=None, bbox_to_anchor=None, fig_filename=None, scenario=0):
    X_all = []
    U_all = []
    labels_all = []

    relevant_keys, constant_keys = get_relevant_keys(settings)
    common_description = get_latex_label_from_setting(get_subdict(settings[0], constant_keys))

    # load
    for i, setting in enumerate(settings):
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True, scenario=scenario)

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
        if results['status'] != 0:
            print(f"Simulation failed with {label}")
        X_all.append(results['x_traj'])
        U_all.append(results['u_traj'])
        # labels_all.append(results['label'])
        labels_all.append(latex_label)

        # doesnt matter when we unpack this
        X_ref = results['x_ref']
        U_ref = results['u_ref']
        # print
        closed_loop_cost = results['cost_total']
        cpu_min = np.min(results['timings']) * 1e3
        cpu_max = np.max(results['timings']) * 1e3
        time_per_iter = 1e3 * np.sum(results['timings']) / np.sum(results['nlp_iter'])
        print(f"Simulation {latex_label}\n\tCPU time: {time_per_iter:.2} ms/iter min {cpu_min:.3} ms, max {cpu_max:.3} ms , closed loop cost: {closed_loop_cost:.5e}")

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
        # title='closed loop' if CLOSED_LOOP else 'open loop',
        idxpx=[0, 1],
        title=title,
        X_ref=X_ref,
        U_ref=U_ref,
        linestyle_list=['--', ':', '--', ':', '--', '-.', '-.', ':'],
        single_column=True,
        xlabel='$t$ [s]',
        idx_xlogy= [4],
        ncol_legend = ncol_legend,
        # color_list=['C0', 'C0', 'C1', 'C1']
        fig_filename=fig_filename,
        bbox_to_anchor=bbox_to_anchor,
    )


def get_results_all_scenarios(setting, latex_label: str) -> dict:
    costs = []
    time_feedback = []
    time_preparation = []
    time_total = []
    res_eq = []
    res_stat = []
    n_fails = 0
    label = get_label_from_setting(setting)
    for scenario in range(N_SCENARIOS):
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True, scenario=scenario)
        # check if file exists
        try:
            with open(results_filename, 'rb') as f:
                results = pickle.load(f)
        except FileNotFoundError:
            print(f"file: for label {latex_label} not found")
            breakpoint()
        if results['status'] != 0:
            print(f"Simulation failed with {label}")
            n_fails += 1
            costs.append(np.nan)
            res_eq.append(np.nan * np.ones(1,))
            res_stat.append(np.nan * np.ones(1,))
            time_feedback.append(np.nan * np.ones(1,))
            time_preparation.append(np.nan * np.ones(1,))
        else:
            costs.append(results['cost_total'])
            res_eq.append(results["res_eq"])
            res_stat.append(results["res_stat"])
            time_feedback.append(results['timings_feedback'])
            time_preparation.append(results['timings_preparation'])
        time_total.append(results['timings'])


    return {"timings_feedback": np.concatenate(time_feedback),
            "timings_preparation": np.concatenate(time_preparation),
            "timings": np.concatenate(time_total),
            "costs": costs,
            "res_eq": np.concatenate(res_eq),
            "res_stat": np.concatenate(res_stat),
            "total_cost": np.sum(costs),
            "n_fails": n_fails,
            }


def get_table_tex_string_from_float(x: float, digits_after_comma=2) -> str:
    digits_after_comma = 2
    if x != x:
        return "NaN"
    if x < 1e4:
        return f"{np.mean(x):.{digits_after_comma}f}"
    else:
        # print in exponential format
        exponent = int(np.floor(np.log10(x)))
        leading_digit = x / 10**exponent
        return f"${leading_digit:.2f} \\cdot 10^" + r"{" + f"{exponent}" + r"}$"

def create_table_as_rti_paper(settings, labels=None, with_reference=True):
    labels_all = []

    relevant_keys, constant_keys = get_relevant_keys(settings)
    common_description = get_latex_label_from_setting(get_subdict(settings[0], constant_keys))

    res_list = []

    # load
    for i, setting in enumerate(settings):

        descriptive_setting = get_subdict(setting, relevant_keys)
        if labels is not None:
            latex_label = labels[i]
        else:
            latex_label = get_latex_label_from_setting(descriptive_setting)

        results_all_scenarios = get_results_all_scenarios(setting, latex_label)
        if results_all_scenarios is not None:
            labels_all.append(latex_label)
            res_list.append(results_all_scenarios)

    # compute relative suboptimality per scenario
    min_cost = np.zeros((N_SCENARIOS,))
    for scenario in range(N_SCENARIOS):
        print(f"{scenario=}")
        min_cost[scenario] = min([result["costs"][scenario] for result in res_list])

    min_total_cost = min(result["total_cost"] for result in res_list)

    if with_reference:
        ref_results = get_results_all_scenarios(REFERENCE_SETTING, 'reference')
        min_total_cost = min(ref_results["total_cost"], min_total_cost)

    for r in res_list:
        r["rel_subopt"] = ((r['total_cost'] - min_total_cost) / min_total_cost) * 100

    for results, setting in zip(res_list, settings):
        rel_subopt = np.zeros(N_SCENARIOS,)
        for scenario in range(N_SCENARIOS):
            if results["costs"][scenario] == min_cost[scenario]:
                rel_subopt[scenario] = 0.
                print(f"{setting['algorithm']}-{setting['nlp_solver_max_iter']} has best cost for scenario {scenario}.")


    # min_cpu = min([1e3 * np.sum(r['timings']) / np.sum(r['nlp_iter']) for r in res_list])
    min_cpu = min([1e3 * np.max(r['timings']) for r in res_list])
    min_time_prep = min([1e3 * np.max(r['timings_preparation']) for r in res_list])
    min_time_feedback = min([1e3 * np.max(r['timings_feedback']) for r in res_list])

    if any([r['n_fails'] > 0 for r in res_list]):
        print("WARNING: some simulations failed")
        with_n_fails = True
    else:
        with_n_fails = False

    n_metrics = 6
    if with_n_fails:
        n_metrics += 1
    with_total_time = False
    if with_total_time:
        n_metrics += 1

    n_settings = len(settings)
    # write table
    table_filename = os.path.join(TEX_FOLDER, "table_clc_benchmark.tex")
    with open(table_filename, 'w') as f:
        f.write(r"\begin{table}" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Timings, relative suboptimality, stationarity residual and constraint violation for different controllers. \label{tab:asrti_experiment}}" + "\n")
        f.write(r"\vspace{-.2cm}")
        f.write(r"\begin{tabular}{l" + n_metrics*"r" + r"}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& \multicolumn{2}{c}{max. timings [ms]} & rel. sub- & \multicolumn{2}{c}{mean}\\" + "\n")
        f.write(r"algorithm & prep & feedback & ")
        if with_n_fails:
            f.write(r"$n\ind{fails}$ &")
        f.write(r" opt. [\%] & $10^3 \norm{g} $ & $\nabla_w\mathcal{L}$ \\ \midrule" + "\n")

        for i_setting, (setting, label, results) in enumerate(zip(settings, labels_all, res_list)):
            line_string = ""
            # line_string += f"{label} &"
            line_string += f"{setting['algorithm']}"
            if setting['algorithm'].startswith('AS-RTI'):
                if setting['algorithm'] != 'AS-RTI-A':
                    line_string += f"-{setting['nlp_solver_max_iter']}"
            elif setting["algorithm"] == "SQP":
                line_string += f"-{setting['nlp_solver_max_iter']}"
            line_string += "&"

            # line_string += f"{setting['N']} &"
            # timing = 1e3 * np.max(results['timings'])
            # if timing > 1.2 * min_cpu:
            #     line_string += f"{timing:.2f} &"
            # else:
            #     line_string += r"\textbf{" + f"{timing:.2f}" + r"} &"

            time_prep = 1e3 * np.max(results['timings_preparation'])
            if time_prep > 1.2 * min_time_prep:
                line_string += f"{time_prep:.2f} &"
            else:
                line_string += r"\textbf{" + f"{time_prep:.2f}" + r"} &"

            time_feedback = 1e3 * np.max(results['timings_feedback'])
            if time_feedback > 1.2 * min_time_feedback:
                line_string += f"{time_feedback:.2f} &"
            else:
                line_string += r"\textbf{" + f"{time_feedback:.2f}" + r"} &"

            if with_n_fails:
                line_string += f"{results['n_fails']} &"

            rel_subopt = results["rel_subopt"]
            float_str = get_table_tex_string_from_float(rel_subopt)
            if rel_subopt < 1.0:
                line_string += r"\textbf{" + float_str + r"}"
            else:
                line_string += float_str
            line_string += r" &"

            res_eq = np.mean(results['res_eq']) * 1e3
            line_string += get_table_tex_string_from_float(res_eq, 0)
            line_string += " &"

            res_stat = np.mean(results['res_stat'])
            line_string += get_table_tex_string_from_float(res_stat)
            line_string += " &"

            line_string += r"\\"
            # if i_setting > 0 and ((i_setting+1) % 3) == 0:
            # if i_setting < n_settings-1 and setting['algorithm'] != settings[i_setting+1]['algorithm']:
            #     line_string += r"\midrule"

            f.write(line_string + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\vspace{-.2cm}")
        f.write(r"\end{table}" + "\n")
    # print table to terminal
    with open(table_filename, 'r') as f:
        print(f.read())
    print(f"saved table as {table_filename}")




def pareto_plot_comparison(settings, with_reference=True, fig_filename='pendulum_pareto_wip.pdf',
        relevant_keys: Optional[list] = None,
        title: Optional[str] = None,
        ncol_legend=None,
        figsize=None,
        performance_indicator='rel_subopt',
                           ):
    print(f"{relevant_keys=}")


    varying_keys, constant_keys = get_relevant_keys(settings)
    if relevant_keys is None:
        relevant_keys = varying_keys
    common_description = get_latex_label_from_setting(get_subdict(settings[0], constant_keys))

    timings = []

    variants = {}
    for k in relevant_keys:
        variants[k] = sorted(set([setting[k] for setting in settings]))
    print(f"{variants=}")

    # sort relevant keys by number of variants
    relevant_keys = sorted(relevant_keys, key=lambda k: len(variants[k]), reverse=True)
    print(f"{relevant_keys=}")
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    alphas = [1.0, 0.5, 0.3]
    markers = ['o', 'v', 's', 'd', '^', '>', '<', 'P', 'D', 'x', 'X']

    i_color = 1
    i_marker = 0
    i_alpha = 2

    # first relevant_key is color, second is alpha, third is marker
    markers_all = []
    colors_all = []
    alphas_all = []
    res_list = []

    # load
    for setting in settings:
        label = get_label_from_setting(setting)

        descriptive_setting = {k: setting[k] for k in relevant_keys}
        latex_label = get_latex_label_from_setting(descriptive_setting)

        # get color, alpha, marker
        results_all_scenarios = get_results_all_scenarios(setting, latex_label)
        if results_all_scenarios is not None:
            res_list.append(results_all_scenarios)
            colors_all.append(colors[variants[relevant_keys[i_color]].index(setting[relevant_keys[i_color]])])
            alphas_all.append(alphas[variants[relevant_keys[i_alpha]].index(setting[relevant_keys[i_alpha]])])
            markers_all.append(markers[variants[relevant_keys[i_marker]].index(setting[relevant_keys[i_marker]])])
        else:
            print(f"skipping setting {setting=}")


    min_total_cost = min(result["total_cost"] for result in res_list)


    # special points and reference
    special_points = []
    special_labels = []
    if with_reference:
        ref_results = get_results_all_scenarios(REFERENCE_SETTING, latex_label)

        ref_cost = ref_results['total_cost']
        ref_timing = np.max(1e3*ref_results['timings'])
        print(f"label {label} got ref_cost {ref_cost:.3e}")
        if performance_indicator == 'rel_subopt':
            special_points.append((((ref_cost - min_total_cost) / min_total_cost) * 100, ref_timing))
        elif performance_indicator == 'res_stat':
            special_points.append((np.mean(ref_results['res_stat']), ref_timing))
        special_labels.append('ideal')


    timings = [np.max(1e3*r['timings']) for r in res_list]
    if performance_indicator == 'rel_subopt':
        xlabel = 'relative suboptimality [\%]'
        rel_subopt = [((r['total_cost'] - min_total_cost) / min_total_cost) * 100 for r in res_list]
        points = list(zip(rel_subopt, timings))
    elif performance_indicator == 'res_stat':
        xlabel = r'mean $\nabla_x \mathcal{L}$'
        res_stat = [np.mean(r['res_stat']) for r in res_list]
        points = list(zip(res_stat, timings))

    color_legend = dict(zip(
            [f'{KEY_TO_TEX[relevant_keys[i_color]]} {v}'
            for v in variants[relevant_keys[i_color]]],
            colors))
    alpha_legend = dict(zip(
        [f'{KEY_TO_TEX[relevant_keys[i_alpha]]} {v}'
          for v in variants[relevant_keys[i_alpha]]],
           alphas))
    marker_legend = dict(zip(
        [f'{KEY_TO_TEX[relevant_keys[i_marker]]} {v}'
          for v in variants[relevant_keys[i_marker]]],
           markers))

    if title is not None:
        title = common_description

    plot_simple_pareto(points, colors_all, alphas_all, markers_all,
                       xlabel=xlabel,
                       ylabel='max computation time [ms]',
                       marker_legend=marker_legend,
                       color_legend=color_legend,
                       alpha_legend=alpha_legend,
                       title=title,
                       special_points=special_points,
                       special_labels=special_labels,
                       fig_filename=fig_filename,
                       ncol_legend=ncol_legend,
                       figsize=figsize
                       )



def plot_residuals_over_time(settings, labels, fig_filename=None, scenario=0):
    from matplotlib import pyplot as plt
    from acados_template import latexify_plot
    latexify_plot()
    fig, axs = plt.subplots(2, 1, figsize=(6, 4.8), sharex=True)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].grid(True)
    axs[1].grid(True)
    markers = ['o', 's', 'd', 'x', 'v', '^', '<', '>', 'p', 'h', 'D', 'P', '*', 'X']
    marker_sizes = [4, 4, 4, 8, 4, 4, 8, 8, 5, 5]
    for setting, plot_label, marker, marker_size in zip(settings, labels, markers, marker_sizes):
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, True, scenario=scenario)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        res_eq = results['res_eq']
        res_stat = results['res_stat']
        axs[0].plot(res_eq, label=plot_label, marker=marker, linestyle='', markersize=marker_size)
        axs[1].plot(res_stat, label=plot_label, marker=marker, linestyle='', markersize=marker_size)
    axs[0].set_xlim([0, len(res_eq)])
    axs[1].set_xlabel('time step')
    axs[0].set_ylabel('infeasibility')
    axs[1].set_ylabel(r'$ || \nabla_x \mathcal{L} ||_\infty$')
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.45, -0.7), ncol=3)

    plt.tight_layout()

    if fig_filename is not None:
        fig_filename = os.path.join("figures", fig_filename)
        plt.savefig(fig_filename, bbox_inches='tight')
        print(f"saved figure as {fig_filename}")
    plt.show()



def main_pareto_plots(settings):
    pareto_plot_comparison(settings, relevant_keys=["algorithm", "nlp_solver_max_iter", "N"],
                           title=None,
                           ncol_legend = 3,
                           figsize=(7.4, 4.6),
                           with_reference=True,
                           fig_filename='pendulum_pareto_subopt.pdf',
                           performance_indicator='rel_subopt')


if __name__ == "__main__":


    SETTINGS = AS_RTI_GRID_SETTINGS + [RTI_SETTING, RTI_2_SETTING, RTI_3_SETTING] + [REFERENCE_SETTING] + [SQP_SETTING]

    run_pendulum_benchmark_closed_loop(SETTINGS)
    main_pareto_plots(SETTINGS)

    SETTING_LABEL_PAIRS = [
        (SQP_SETTING, "SQP"),
        # (REFERENCE_SETTING, "reference"),
        # (RTI_3_SETTING, "RTI-3"),
        (RTI_2_SETTING, "RTI-2"),
        # (AS_RTI_D10_SETTING, "AS-RTI-D-10"),
        (AS_RTI_2_SETTING, "AS-RTI-D-2"),
        (AS_RTI_SETTING, "AS-RTI-D-1"),
        # (AS_RTI_C10_SETTING, 'AS-RTI-C-10'),
        # (AS_RTI_C5_SETTING, 'AS-RTI-C-5'),
        # (AS_RTI_C8_SETTING, 'AS-RTI-C-8'),
        # (AS_RTI_C3_SETTING, 'AS-RTI-C-3'),
        (AS_RTI_C2_SETTING, 'AS-RTI-C-2'),
        (AS_RTI_C1_SETTING, 'AS-RTI-C-1'),
        # (AS_RTI_B10_SETTING, "AS-RTI-B-10"),
        (AS_RTI_B2_SETTING, "AS-RTI-B-2"),
        (AS_RTI_B1_SETTING, "AS-RTI-B-1"),
        (AS_RTI_A_SETTING, "AS-RTI-A"),
        (RTI_SETTING, "RTI"),
        ]
    SETTINGS = [setting for setting, label in SETTING_LABEL_PAIRS]
    LABELS = [label for setting, label in SETTING_LABEL_PAIRS]
    create_table_as_rti_paper(SETTINGS, LABELS)

    SETTING_LABEL_PAIRS = [
        (SQP_SETTING, "SQP"),
        # (RTI_2_SETTING, "RTI-2"),
        (AS_RTI_D10_SETTING, "AS-RTI-D-10"),
        (AS_RTI_SETTING, "AS-RTI-D-1"),
        (AS_RTI_C10_SETTING, 'AS-RTI-C-10'),
        (AS_RTI_C1_SETTING, 'AS-RTI-C-1'),
        # (AS_RTI_C3_SETTING, 'AS-RTI-C-3'),
        # (AS_RTI_C2_SETTING, 'AS-RTI-C-2'),
        # (AS_RTI_B10_SETTING, "AS-RTI-B-10"),
        (AS_RTI_B1_SETTING, "AS-RTI-B-1"),
        (AS_RTI_A_SETTING, "AS-RTI-A"),
        (RTI_SETTING, "RTI"),
        ]

    SETTINGS = [setting for setting, label in SETTING_LABEL_PAIRS]
    LABELS = [label for setting, label in SETTING_LABEL_PAIRS]

    # for scenario in range(N_SCENARIOS): #[69, 99]:
    #     plot_trajectories(SETTINGS,
    #                     labels = LABELS,
    #                     ncol_legend=3,
    #                     title="",
    #                     bbox_to_anchor=(0.5, -0.85),
    #                     fig_filename=f'pendulum_trajectories_as_rti_{scenario}.pdf',
    #                     scenario=scenario
    #                 )

