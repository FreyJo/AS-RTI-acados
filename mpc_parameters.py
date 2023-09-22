import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class MpcParameters:
    umin: np.ndarray  # lower bound on u
    umax: np.ndarray  # upper bound on u
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    us: np.ndarray
    xs: np.ndarray
    # Tf: float = 0.25 * 13  # horizon length
    N: int
    dt: float
    linear_mpc: bool = False
    cost_integration: bool = False
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    sim_method_num_steps_0: int = 1
    lbx: Optional[np.ndarray] = None
    ubx: Optional[np.ndarray] = None
    idxbx: Optional[np.ndarray] = None
    gamma_penalty: float = 1.0


@dataclass
class MpcPendulumParameters(MpcParameters):

    def __init__(self, xs, us):
        self.Q = 2*np.diag([1e2, 1e3, 1e-2, 1e-2])
        # self.R = 2*np.diag([1e-0])
        self.R = 2*np.diag([2e-1])
        self.us = us
        self.xs = xs
        self.umin = -np.array([40.0])
        self.umax = np.array([40.0])
        #
        # NOTE: computed with setup_linearized_model()
        self.P = np.array([[ 215.86822404, -294.53515325,  114.32904314,  -98.65929707],
                           [-294.53515325,  872.1412253 , -216.25863682,  216.67703862],
                           [ 114.32904314, -216.25863682,   92.23297629,  -81.47144968],
                           [ -98.65929707,  216.67703862,  -81.47144968,   75.74192616]]

            )
        self.N = 20
        self.dt = 0.05
        self.T = 2
        self.sim_method_num_steps = 1
        self.sim_method_num_steps_0 = 1
        # constraints
        # self.lbx = np.array([-1.0])
        # self.ubx = np.array([1.0])
        # self.idxbx = np.array([0])
        # self.gamma_penalty = 1e5


