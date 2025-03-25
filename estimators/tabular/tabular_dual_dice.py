# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.tabular_dice import TabularDice
from dice_rl_TU_Vienna.utils.numpy import project_in, project_out, safe_divide

# ---------------------------------------------------------------- #

class TabularDualDice(TabularDice):
    @property
    def __name__(self): return "TabularDualDice"

    def solve_sdc(self, gamma, **kwargs):

        projected = kwargs["projected"]

        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        mask = dD_bar == 0
        (d0_, dD_), (P_,) = project_in(mask, (d0_bar, dD_bar), (P_bar,), projected)

        # -------------------------------- #

        D_ = np.diag(dD_)

        sqrt_dD_ = np.sqrt(dD_)

        A = safe_divide(
            D_ - gamma * P_.T,
            sqrt_dD_,
            zero_div_zero=-1)

        a = np.dot(A, A.T)
        b = (1 - gamma) * d0_

        hv_ = np.linalg.solve(a, b)
        sdc_ = safe_divide(np.dot(hv_, A), sqrt_dD_, zero_div_zero=-1)

        # -------------------------------- #

        hv_hat  = project_out(mask, hv_,  projected, masking_value=-1)
        sdc_hat = project_out(mask, sdc_, projected, masking_value=-1)

        info = { "hv_hat": hv_hat, }

        return sdc_hat, info

# ---------------------------------------------------------------- #
