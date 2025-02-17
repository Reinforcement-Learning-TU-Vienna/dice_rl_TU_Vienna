# ---------------------------------------------------------------- #

import numpy as np

from utils.numpy import safe_divide

# ---------------------------------------------------------------- #

def get_error_auxiliary_estimates(auxiliary_estimates, analytical_solver):

    d0_bar, dD_bar, P_bar, r_bar, n = auxiliary_estimates

    d0 = analytical_solver.d0
    dD = analytical_solver.dD
    P  = analytical_solver.P

    sizes = {
        "d0": np.sum(d0  != 0),
        "dD": np.sum(dD  != 0),
        "P":  np.sum(P.T != 0),
    }

    delta = {
        "d0": d0_bar / n - d0,
        "dD": dD_bar  / n - dD,
        "P":  safe_divide(P_bar.T, dD_bar) - P.T,
    }

    errors = {
        "sum":    { k: np.sum (v**2)            for k, v in delta.items() },
        "mean":   { k: np.mean(v**2)            for k, v in delta.items() },
        "masked": { k: np.sum (v**2) / sizes[k] for k, v in delta.items() },
    }

    return errors

# ---------------------------------------------------------------- #