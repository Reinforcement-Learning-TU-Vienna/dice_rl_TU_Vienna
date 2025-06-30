# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.model_based_tabular_OffPE import ModelBasedTabularOffPE
from dice_rl_TU_Vienna.estimators.tabular.bellman_equations import solve_forward_bellman_equations_approximate

# ---------------------------------------------------------------- #

class TabularVafe(ModelBasedTabularOffPE):
    @property
    def __name__(self): return "TabularVafe"

    def solve_vaf(self, gamma, **kwargs):

        projected = kwargs["projected"]

        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        Q_hat, info = solve_forward_bellman_equations_approximate(
            dD_bar, r_bar, P_bar, gamma, projected, )

        return Q_hat, info

    def solve_pv(self, gamma, Q_hat, info):

        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        if gamma < 1:
            d0_hat = d0_bar / n
            rho_hat = (1 - gamma) * np.dot(Q_hat, d0_hat)

        else:
            rho_hat = info["pv_approx"]

        return rho_hat

    def solve(self, gamma, **kwargs):

        Q_hat, info = self.solve_vaf(gamma, **kwargs)
        rho_hat = self.solve_pv(gamma, Q_hat, info)

        info["Q_hat"] = Q_hat

        return rho_hat, info

# ---------------------------------------------------------------- #
