import casadi as ca
import numpy as np

from dataclasses import dataclass
from acados_template import AcadosModel

from utils import compute_lqr_gain

from typing import Tuple, Optional

@dataclass
class ModelParameters:
    nx_original: int = None
    nu_original: int = None
    xs: np.ndarray = None
    us: np.ndarray = None
    parameter_values : np.ndarray = None
    cost_state_idx: int = None
    cost_state_dyn_fun = None
    xlabels: Optional[list] = None
    ulabels: Optional[list] = None



def setup_linearized_model(model, model_params, mpc_params):
    # linearized dynamics
    A, B, P = compute_lqr_gain(model, model_params, mpc_params)
    f_discrete = (
        model_params.xs + A @ (model.x - model_params.xs) + B @ (model.u - model_params.us)
    )

    model.disc_dyn_expr = f_discrete

    print(f"P_mat {P}")
    return model, model_params


def setup_pendulum_model() -> Tuple[AcadosModel, ModelParameters]:

    model_name = 'pendulum'

    # constants
    M = 1. # mass of the cart [kg] -> now estimated
    m = 0.1 # mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    l = 0.8 # length of the rod [m]

    # set up states & controls
    x1      = ca.SX.sym('x1')
    theta   = ca.SX.sym('theta')
    v1      = ca.SX.sym('v1')
    dtheta  = ca.SX.sym('dtheta')

    x = ca.vertcat(x1, theta, v1, dtheta)

    F = ca.SX.sym('F')
    u = ca.vertcat(F)

    # xdot
    x1_dot      = ca.SX.sym('x1_dot')
    theta_dot   = ca.SX.sym('theta_dot')
    v1_dot      = ca.SX.sym('v1_dot')
    dtheta_dot  = ca.SX.sym('dtheta_dot')

    xdot = ca.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # parameters
    p = ca.SX.sym('p', 0)

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m*cos_theta*cos_theta
    f_expl = ca.vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    # cost
    model.cost_y_expr = ca.vertcat(x, u)
    model.cost_y_expr_e = x

    # Model params
    model_params = ModelParameters(
        nx_original = 4,
        nu_original = 1,
        xs = np.array([0.0, 0.0, 0.0, 0.0]),
        us = np.array([0.0]),
        parameter_values= np.array([]),
        xlabels = ['x', 'theta', 'v', 'dtheta'],
        ulabels = ['F'],
    )

    return model, model_params

