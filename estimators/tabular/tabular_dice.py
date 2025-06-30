# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.model_based_tabular_OffPE import ModelBasedTabularOffPE
from dice_rl_TU_Vienna.estimators.tabular.bellman_equations import (
    solve_backward_bellman_equations_approximate, )

# ---------------------------------------------------------------- #

class TabularDice(ModelBasedTabularOffPE):
    @property
    def __name__(self): return "TabularDice"

    def solve_sdc(self, gamma, **kwargs):

        projected = kwargs["projected"]
        modified  = kwargs["modified"]

        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        w_hat, info = solve_backward_bellman_equations_approximate(
            d0_bar, dD_bar, P_bar, n, gamma, modified, projected, )

        return w_hat, info

    def solve_pv(self, w_hat, weighted):
        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        a = np.dot(r_bar, w_hat)
        b = n if not weighted else np.dot(dD_bar, w_hat)

        rho_hat = a / b

        return rho_hat

    def solve(self, gamma, **kwargs):

        weighted = kwargs["weighted"]

        w_hat, info = self.solve_sdc(gamma, **kwargs)
        rho_hat = self.solve_pv(w_hat, weighted)

        info["w_hat"] = w_hat

        return rho_hat, info

# ---------------------------------------------------------------- #
