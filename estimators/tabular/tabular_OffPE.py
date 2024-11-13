# ---------------------------------------------------------------- #

from abc import ABC, abstractmethod

from dice_rl_TU_Vienna.estimators.tabular.utils import get_dim, obs_act_to_index

# ---------------------------------------------------------------- #

class TabularOffPE(ABC):
    @property
    @abstractmethod
    def __name__(self): pass

    def __init__(self, aux_estimates, num_obs, n_act, obs_act=True):
        self.aux_estimates = aux_estimates
        self.num_obs = num_obs
        self.n_act = n_act
        self.obs_act = obs_act

        self.dim = get_dim(num_obs, n_act, obs_act)

    def get_index(self, obs, act):
        return obs_act_to_index(
            obs, act,
            self.num_obs, self.n_act,
            self.obs_act, neighbours="act", )


# ---------------------------------------------------------------- #
