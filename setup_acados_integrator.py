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


from acados_template import AcadosSim, AcadosSimSolver, casadi_length
import numpy as np


def setup_acados_integrator(
    model,
    dt,
    mpc_params,
    sensitivity_propagation=False,
    num_stages=4,
    num_steps=100,
    integrator_type="IRK",
    collocation_type = "GAUSS_RADAU_IIA",
):

    sim = AcadosSim()

    # set model
    sim.model = model

    # set simulation time
    sim.solver_options.T = dt

    ## set options
    sim.solver_options.integrator_type = integrator_type
    sim.solver_options.num_stages = num_stages
    sim.solver_options.num_steps = num_steps
    # for implicit integrator
    sim.solver_options.newton_iter = 20
    sim.solver_options.newton_tol = 1e-14
    sim.solver_options.collocation_type = collocation_type
    # sensitivity_propagation
    sim.solver_options.sens_adj = sensitivity_propagation
    sim.solver_options.sens_forw = sensitivity_propagation
    sim.solver_options.sens_hess = sensitivity_propagation

    # nominal parameter values
    # TODO
    if casadi_length(model.p) > 0:
        sim.parameter_values = np.concatenate((mpc_params.xs, mpc_params.us))

    # create
    acados_integrator = AcadosSimSolver(sim, verbose=False)

    return acados_integrator
