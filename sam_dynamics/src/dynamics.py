# General class for dynamics
# Use, e.g., for optimal control, MPC, etc.
# Christopher Iliffe Sprague

import jax.numpy as np
from jax import jit, jacfwd
import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Dynamics:

    def __init__(self, **kwargs):

        # constant parameters
        assert all(key in self.params.keys() for key in kwargs.keys())
        self.params.update(kwargs)

        # compute and compile Jacobian
        self.eom_jac = jit(jacfwd(self.eom))

    def eom(self, state, control, *args):
        raise NotImplementedError

    def propagate(self, state, controller, t0, tf, atol=1e-8, rtol=1e-8, method='DOP853'):

        # integrate dynamics
        sol = solve_ivp(
            jit(lambda t, x: self.eom(x, controller(x), *self.params.values())),
            (t0, tf),
            state,
            method=method,
            rtol=rtol,
            atol=atol,
            jac=jit(lambda t, x: self.eom_jac(x, controller(x), *self.params.values()))
        )

        # return times, states, and controls
        times, states = sol.t, sol.y.T
        controls = np.apply_along_axis(controller, 1, states)
        return times, states, controls

    def plot(self, states, controls=None):
        raise NotImplementedError
