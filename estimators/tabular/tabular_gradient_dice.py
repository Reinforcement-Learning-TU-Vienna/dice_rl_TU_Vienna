# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.tabular_dice import TabularDice
from dice_rl_TU_Vienna.utils.numpy import project_in, project_out, safe_divide

# ---------------------------------------------------------------- #

class TabularGradientDice(TabularDice):
    @property
    def __name__(self): return "TabularGradientDice"

    def solve_sdc(self, gamma, **kwargs):

        projected = kwargs["projected"]
        lamda = kwargs["lamda"]

        d0_bar, dD_bar, P_bar, r_bar, n = self.auxiliary_estimates.bar

        mask = dD_bar == 0
        (d0_, dD_), (P_,) = project_in(mask, (d0_bar, dD_bar), (P_bar,), projected)

        # -------------------------------- #

        D_ = np.diag(dD_)

        A = D_ - gamma * P_.T
        B = safe_divide(A.T, dD_, zero_div_zero=-1)

        x = np.matmul(B, A)
        y = lamda / n * np.outer(d0_, d0_)
        a = x + y

        x = (1 - gamma) * np.matmul(B, d0_)
        y = lamda * d0_
        b = x + y

        sdc_ = np.linalg.solve(a, b)

        # -------------------------------- #

        sdc_hat = project_out(mask, sdc_, projected, masking_value=-1)

        info = {}

        return sdc_hat, info

# ---------------------------------------------------------------- #
