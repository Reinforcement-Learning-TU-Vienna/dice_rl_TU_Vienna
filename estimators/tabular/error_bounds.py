# ---------------------------------------------------------------- #

import numpy as np

# ---------------------------------------------------------------- #

def get_error_bound_exact(w_hat, gamma, analytical_solver):

    d0 = analytical_solver.initial_distribution
    d  = analytical_solver.stationary_distribution_dataset
    P  = analytical_solver.transition_matrix

    a = (1 - gamma) * d0
    b = gamma * P @ (d * w_hat)
    c = d * w_hat

    x = np.linalg.norm(a + b - c, ord=1)
    y = 1 / (1 - gamma)

    return x * y

def get_error_bound_approximate(w_hat, gamma, auxiliary_estimates):

    d0_bar, dD_bar, P_bar, r_bar, n = auxiliary_estimates

    a = (1 - gamma) * d0_bar
    b = gamma * P_bar.T @ w_hat
    c = dD_bar * w_hat

    x = np.linalg.norm(a + b - c, ord=1)
    y = 1 / (1 - gamma)
    z = 1 / n

    return x * y * z

# ---------------------------------------------------------------- #
