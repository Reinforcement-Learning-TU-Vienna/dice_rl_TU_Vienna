# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.tabular_OffPE import TabularOffPE
from dice_rl_TU_Vienna.estimators.tabular.bellman_equations import solve_forwards_bellman_equations_approximate

# ---------------------------------------------------------------- #

class TabularVafe(TabularOffPE):
    @property
    def __name__(self): return "TabularVafe"

    def solve_vaf(self, gamma, projected, **kwargs):

        d0_bar, dD_bar, P_bar, r_bar, n = self.aux_estimates

        vf_hat, info = solve_forwards_bellman_equations_approximate(
            dD_bar, r_bar, P_bar, gamma, projected, )

        return vf_hat, info

    def solve_pv(self, gamma, vf_hat, info):

        d0_bar, dD_bar, P_bar, r_bar, n = self.aux_estimates

        if gamma < 1:
            d0_hat = d0_bar / n
            pv_hat = (1 - gamma) * np.dot(vf_hat, d0_hat)

        else:
            pv_hat = info["pv_hat"]

        return pv_hat

    def solve(self, gamma, projected, **kwargs):

        vf_hat, info = self.solve_vaf(gamma, projected, **kwargs)
        pv_hat = self.solve_pv(gamma, vf_hat, info)

        return pv_hat, vf_hat, info

# ---------------------------------------------------------------- #
