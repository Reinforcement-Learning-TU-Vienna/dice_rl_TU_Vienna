# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.estimators.onpolicy import OnPE

# ---------------------------------------------------------------- #

class IS(OnPE):
    @property
    def __name__(self): return "IS"

    @OnPE.dataset.setter
    def dataset(self, dataset):
        OnPE.dataset.__set__(self, dataset)

        self.q = np.array(
                dataset.apply(
                lambda x: x["probs"][ x["act"] ] / x["probs_behavior"][ x["act"] ],
                axis=1,
            )
        )

        x = np.cumsum(self.H)
        x = np.insert(x, 0, 0)
        x = x[:-1]
        self.W = np.multiply.reduceat(self.q, x)

    def solve(self, gamma, **kwargs):
    
        weighted = kwargs["weighted"]

        kwargs["W"] = np.repeat(self.W, self.H)
        kwargs["m"] = self.m if not weighted else np.sum(self.W)

        return super().solve(gamma, **kwargs)

# ---------------------------------------------------------------- #
