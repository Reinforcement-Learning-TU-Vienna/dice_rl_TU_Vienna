# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

import warnings

from dice_rl.data.dataset import convert_to_tfagents_timestep
from dice_rl_TU_Vienna.estimators.estimator import get_fullbatch_average
from dice_rl_TU_Vienna.estimators.tabular.utils import get_num_obs_n_act
from dice_rl_TU_Vienna.estimators.tabular.tabular_OffPE import TabularOffPE
from dice_rl_TU_Vienna.estimators.tabular.bellman_equations import solve_backwards_bellman_equations_approximate

# ---------------------------------------------------------------- #

class TabularDice(TabularOffPE):
    @property
    def __name__(self): return "TabularDice"

    def solve_sdc(self, gamma, projected, **kwargs):

        modified = kwargs["modified"]

        d0_bar, dD_bar, P_bar, r_bar, n = self.aux_estimates

        sdc_hat, info = solve_backwards_bellman_equations_approximate(
            d0_bar, dD_bar, P_bar, n, gamma, modified, projected, )

        return sdc_hat, info

    def solve_pv(self, sdc_hat, weighted):

        def weight_fn(env_step):
            index = self.get_index(env_step.observation, env_step.action)
            dual = sdc_hat[index] # type: ignore

            if not np.prod(dual >= 0):
                warnings.warn("negative sdc value encountered", UserWarning)

            policy_ratio = 1.0
            if not self.obs_act:
                tfagents_timestep = convert_to_tfagents_timestep(env_step)

                target_log_probabilities = self.evaluation_policy \
                    .distribution(tfagents_timestep) \
                    .action \
                    .log_prob(env_step.action) # type: ignore

                x = target_log_probabilities
                y = env_step.get_log_probability()
                policy_ratio = tf.exp(x - y)

            return tf.cast(dual * policy_ratio, tf.float32)

        pv_hat = get_fullbatch_average(
            dataset=self.dataset, by=self.by, weight_fn=weight_fn, weighted=weighted, )

        return pv_hat

    def solve(self, gamma, projected, **kwargs):

        weighted = kwargs["weighted"]

        sdc_hat, info = self.solve_sdc(gamma, projected, **kwargs)
        pv_hat = self.solve_pv(sdc_hat, weighted)

        return pv_hat, sdc_hat, info

    def __init__(
            self,
            dataset, evaluation_policy,
            aux_estimates,
            obs_act=True, by="steps"):

        self.dataset = dataset
        self.evaluation_policy = evaluation_policy
        self.by = by

        num_obs, n_act = get_num_obs_n_act(
            self.dataset.spec, obs_act )

        super().__init__(aux_estimates, num_obs, n_act, obs_act)

# ---------------------------------------------------------------- #
