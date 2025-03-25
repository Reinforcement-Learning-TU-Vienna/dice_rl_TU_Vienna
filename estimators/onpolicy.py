# ---------------------------------------------------------------- #

import numpy as np

# ---------------------------------------------------------------- #

class OnPE:
    @property
    def __name__(self): return "OnPE"

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        dataset = dataset.sort_values(by=["id", "t"])
        self._dataset = dataset

        self.r = dataset["rew"]
        self.t = dataset["t"]
        self.m = dataset["id"].unique().size
        _, self.H = np.unique(dataset["id"], return_counts=True)

    def solve(self, gamma, **kwargs):

        scale = kwargs.get("scale", True)
        W = kwargs.get("W", 1)
        y = kwargs.get("m", self.m)

        if gamma < 1:
            f = (1 - gamma) if scale else 1
            x = f * np.sum( self.r * gamma ** self.t * W )

        else:
            f = 1 / np.repeat(self.H, self.H) if scale else 1
            x = np.sum( f * self.r * W )

        rho_hat = x / y
        info = {}

        return rho_hat, info

# ---------------------------------------------------------------- #
