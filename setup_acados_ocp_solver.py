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


from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, casadi_length
from scipy.linalg import block_diag
import numpy as np
import casadi as ca
from casadi import vertcat


from models import ModelParameters
from utils import get_nbx_violation_expression
from mpc_parameters import MpcParameters

OCP_SOLVER_NUMBER = 0

def augment_model_with_clock_state(model: AcadosModel):

    t = ca.SX.sym('t')
    tdot = ca.SX.sym('tdot')

    model.x = ca.vertcat(model.x, t)
    model.xdot = ca.vertcat(model.xdot, tdot)
    model.f_expl_expr = ca.vertcat(model.f_expl_expr, 1)
    model.f_impl_expr = model.f_expl_expr - model.xdot

    model.clock_state = t

    return model


def augment_model_with_cost_state(model: AcadosModel, params: ModelParameters, mpc_params: MpcParameters):

    cost_state = ca.SX.sym('cost_state')
    cost_state_dot = ca.SX.sym('cost_state_dot')

    x_ref = ca.SX.sym('x_ref', params.nx_original)
    u_ref = ca.SX.sym('u_ref', params.nu_original)
    xdiff = x_ref - model.x[:params.nx_original]
    udiff = u_ref - model.u[:params.nu_original]

    cost_state_dyn = .5 * (xdiff.T @ mpc_params.Q @xdiff + udiff.T @mpc_params.R @udiff)
    if mpc_params.lbx is not None:
        nbx = mpc_params.lbx.size
        # formulate as cost!
        x = model.x
        violation_expr = get_nbx_violation_expression(model, mpc_params)
        cost_state_dyn += 0.5 * (violation_expr.T @ mpc_params.gamma_penalty*np.eye(nbx) @ violation_expr)
    if hasattr(model, 'barrier'):
        cost_state_dyn += model.barrier

    model.x = ca.vertcat(model.x, cost_state)
    params.cost_state_idx = casadi_length(model.x) - 1
    params.cost_state_dyn_fun = ca.Function('cost_state_dyn_fun', [model.x, model.u, x_ref, u_ref], [cost_state_dyn])
    model.xdot = ca.vertcat(model.xdot, cost_state_dot)
    model.f_expl_expr = ca.vertcat(model.f_expl_expr, cost_state_dyn)
    model.f_impl_expr = model.f_expl_expr - model.xdot
    model.p = ca.vertcat(model.p, x_ref, u_ref)

    params.parameter_values = np.concatenate((params.parameter_values, np.zeros(params.nx_original + params.nu_original)))
    params.xs = np.append(params.xs, [0.0])

    params.xlabels = params.xlabels + ['cost_state']

    return model



def augment_model_with_picewise_linear_u(model: AcadosModel, model_params: ModelParameters, mpc_params: MpcParameters):

    model = augment_model_with_clock_state(model)
    nu = casadi_length(model.u)
    # new controls
    u_0 = ca.SX.sym('u_0', nu)
    u_1 = ca.SX.sym('u_1', nu)
    mpc_params.umin = np.concatenate((mpc_params.umin, mpc_params.umin))
    mpc_params.umax = np.concatenate((mpc_params.umax, mpc_params.umax))
    # parameters
    clock_state0 = ca.SX.sym('clock_state0', 1)
    delta_t_n = ca.SX.sym('delta_t_n', 1)
    model.p = ca.vertcat(model.p, clock_state0, delta_t_n)

    tau = (model.clock_state - clock_state0) / delta_t_n

    u_pwlin = u_0 * (1 - tau) + u_1 * tau
    model.f_expl_expr = ca.substitute(model.f_expl_expr, model.u, u_pwlin)
    model.f_impl_expr = ca.substitute(model.f_impl_expr, model.u, u_pwlin)
    model.cost_y_expr = ca.substitute(model.cost_y_expr, model.u, u_pwlin)

    model_params.xs = np.append(model_params.xs, [0.0])
    model_params.us = np.concatenate((model_params.us, model_params.us))
    model_params.parameter_values = np.concatenate((model_params.parameter_values, [0.0, 0.0]))

    model.u = ca.vertcat(u_0, u_1)
    return model, model_params, mpc_params


def setup_acados_ocp_without_options(model: AcadosModel, model_params: ModelParameters, mpc_params: MpcParameters) -> AcadosOcp:

    ocp = AcadosOcp()

    # set model
    ocp.model = model
    x = model.x
    u = model.u
    nx = x.shape[0]
    nu = u.shape[0]

    # set cost
    ocp.cost.W_e = mpc_params.P
    ocp.cost.W = block_diag(mpc_params.Q, mpc_params.R)
    if model.cost_y_expr is not None:
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.yref = np.zeros((casadi_length(ocp.model.cost_y_expr),))
        ocp.cost.yref = np.concatenate((model_params.xs, model_params.us))
    else:
        ocp.cost.cost_type = "EXTERNAL"

    if model.cost_y_expr_e is not None:
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.yref_e = np.zeros((casadi_length(ocp.model.cost_y_expr_e),))
        ocp.cost.yref_e = model_params.xs
    else:
        ocp.cost.cost_type_e = "EXTERNAL"

    nx = casadi_length(model.x)
    nxs = model_params.xs.size
    ocp.constraints.x0 = np.hstack((model_params.xs, np.zeros(nx-nxs)))

    # set constraints
    if mpc_params.umin is not None:
        ocp.constraints.lbu = mpc_params.umin
        ocp.constraints.ubu = mpc_params.umax
        ocp.constraints.idxbu = np.arange(nu)

    # formulate constraint violation as cost!
    if mpc_params.lbx is not None and ocp.cost.cost_type != "EXTERNAL":
        nbx = mpc_params.lbx.size
        # ocp.constraints.lbx = mpc_params.lbx
        # ocp.constraints.ubx = mpc_params.ubx
        # ocp.constraints.idxbx = mpc_params.idxbx

        violation_expression = get_nbx_violation_expression(model, mpc_params)
        ocp.model.cost_y_expr = vertcat(ocp.model.cost_y_expr, violation_expression)
        ocp.cost.yref = np.concatenate((ocp.cost.yref, np.zeros((nbx))))
        ocp.cost.W = block_diag(ocp.cost.W, mpc_params.gamma_penalty*np.eye(nbx))

        # ocp.model.cost_y_expr = vertcat(violation_expression, ocp.model.cost_y_expr)
        # ocp.cost.yref = np.concatenate((np.zeros((nbx)), ocp.cost.yref))
        # ocp.cost.W = block_diag(mpc_params.gamma_penalty*np.eye(nbx), ocp.cost.W)

        ocp.model.cost_y_expr_e = vertcat(ocp.model.cost_y_expr_e, violation_expression)
        ocp.cost.yref_e = np.concatenate((ocp.cost.yref_e, np.zeros((nbx))))
        ocp.cost.W_e = block_diag(ocp.cost.W_e, mpc_params.gamma_penalty*np.eye(nbx))

    ocp.translate_nls_cost_to_conl()

    if isinstance(model.p, list):
        ocp.parameter_values = np.zeros(0)
    else:
        ocp.parameter_values = np.zeros(model.p.rows())

    return ocp


def setup_acados_ocp_solver(
    model: AcadosModel, model_params, mpc_params: MpcParameters, use_rti=False,
    nlp_tol=1e-6,
    levenberg_marquardt=1e-4,
    nlp_solver_max_iter=100,
    hessian_approx='GAUSS_NEWTON',
    regularize_method = 'NO_REGULARIZE',
    algorithm='SQP',
    integrator_type = "IRK",
    rti_log_residuals=0
):

    ocp = setup_acados_ocp_without_options(model, model_params, mpc_params)

    # set options
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_DAQP"
    # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = mpc_params.N  # for partial condensing

    # number of shooting intervals
    if isinstance(ocp, AcadosOcp):
        ocp.dims.N = mpc_params.N
    N_horizon = mpc_params.N
    dt = mpc_params.dt
    # ocp.solver_options.time_steps = np.array([dt]+ (N_horizon-1)*[(mpc_params.T - dt)/ (N_horizon-1)])

    time_steps = np.zeros((N_horizon, ))
    time_steps[0] = dt
    remaining_time = mpc_params.T - dt
    for i_step in range(1, N_horizon-1):
        dt_multiple = max(1,
                np.floor(
                    ((remaining_time) /(N_horizon-i_step))/ dt)
        )
        if dt_multiple < 1:
            breakpoint()
        time_steps[i_step] = dt_multiple * dt
        remaining_time -= time_steps[i_step]
    if remaining_time < 0:
        breakpoint()
    time_steps[-1] = remaining_time
    ocp.solver_options.time_steps = time_steps

    # set prediction horizon
    ocp.solver_options.tf = sum(ocp.solver_options.time_steps)

    ocp.solver_options.hessian_approx = hessian_approx
    if algorithm == "RTI":
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
    elif algorithm == "SQP":
        ocp.solver_options.nlp_solver_type = "SQP"
    elif algorithm == "AS-RTI-py":
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
    elif algorithm in ["AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.as_rti_iter = nlp_solver_max_iter
    else:
        raise NotImplementedError()

    ocp.solver_options.rti_log_residuals = rti_log_residuals

    ocp.solver_options.integrator_type = integrator_type
    ocp.solver_options.sim_method_num_stages = mpc_params.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps = np.array([mpc_params.sim_method_num_steps_0] + (mpc_params.N-1)*[mpc_params.sim_method_num_steps])
    ocp.solver_options.sim_method_newton_iter = 3
    # ocp.solver_options.sim_method_newton_tol = 1e-6
    ocp.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    # ocp.solver_options.collocation_type = "EXPLICIT_RUNGE_KUTTA"

    if mpc_params.cost_integration:
        ocp.solver_options.cost_discretization = 'INTEGRATOR'

    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.levenberg_marquardt = levenberg_marquardt
    ocp.solver_options.tol = nlp_tol
    ocp.solver_options.qp_tol = 1e-1 * nlp_tol
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    # ocp.solver_options.print_level = 1
    # ocp.solver_options.nlp_solver_ext_qp_res = 1

    ocp.solver_options.regularize_method = regularize_method
    ocp.solver_options.reg_epsilon = 1e-8
    if hessian_approx == 'EXACT':
        ocp.solver_options.regularize_method = 'PROJECT'

    # create
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)

    return ocp_solver


