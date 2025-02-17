# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.tabular_OffPE import TabularOffPE
from dice_rl_TU_Vienna.estimators.tabular.bellman_equations import (
    solve_backwards_bellman_equations_approximate, )

# ---------------------------------------------------------------- #

class TabularDice(TabularOffPE):
    @property
    def __name__(self): return "TabularDice"

    def solve_sdc(self, gamma, projected, **kwargs):

        modified = kwargs["modified"]

        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        w_hat, info = solve_backwards_bellman_equations_approximate(
            d0_bar, dD_bar, P_bar, n, gamma, modified, projected, )

        return w_hat, info

    def solve_pv(self, w_hat, weighted):
        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        a = np.sum(r_bar * w_hat)
        b = n if weighted else np.sum(dD_bar * w_hat)

        rho_hat = a / b

        return rho_hat

    def solve(self, gamma, projected, **kwargs):

        weighted = kwargs["weighted"]

        w_hat, info = self.solve_sdc(gamma, projected, **kwargs)
        rho_hat = self.solve_pv(w_hat, weighted)

        return rho_hat, w_hat, info

# ---------------------------------------------------------------- #
