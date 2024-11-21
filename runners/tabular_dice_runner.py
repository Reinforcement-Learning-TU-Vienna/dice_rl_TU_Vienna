# ---------------------------------------------------------------- #

import os

from dice_rl_TU_Vienna.dataset import load_or_create_dataset_Dataframe
from dice_rl_TU_Vienna.specs import get_observation_action_spec_tabular

from dice_rl_TU_Vienna.estimators.tabular.aux_estimates.io import load_or_create_aux_estimates
from dice_rl_TU_Vienna.estimators.tabular.tabular_dice          import TabularDice
from dice_rl_TU_Vienna.estimators.tabular.tabular_dual_dice     import TabularDualDice
from dice_rl_TU_Vienna.estimators.tabular.tabular_gradient_dice import TabularGradientDice

# ---------------------------------------------------------------- #

class TabularDiceRunner_Dataframe(object):
    estimators_dict = {
        "TabularDice": TabularDice,
        "TabularDualDice": TabularDualDice,
        "TabularGradientDice": TabularGradientDice,
    }

    def __init__(self, df, bounds, get_split, get_episode, base_dir, sub_dir=None):
        """
        Args:
            df: pandas.DataFrame, for example with columns
                - "id": int, # episode identifier
                - "t": int, # time step
                - "obs": int, # observation or state
                - "act": int, # action
                - "rew": float, # reward
                - "probs_eval": float, # probability of action act in state obs under evaluation policy
            bounds: (obs_min, obs_max, act_min, act_max)
            get_split: (df) -> (id, obs, act, rew, probs_eval)
            get_episode: (df, id) -> df[where df_id matches id]
        """

        self.df = df
        self.bounds = bounds
        self.get_split = get_split
        self.get_episode = get_episode
        self.base_dir = base_dir
        self.sub_dir = sub_dir

        self.dataset = None
        self.evaluation_policy = None
        self.aux_estimates = None

    @property
    def datasets_dir(self):
        dir = os.path.join(self.base_dir, "datasets")
        if self.sub_dir is not None: dir = os.path.join(dir, self.sub_dir)
        return dir

    @property
    def outputs_dir(self):
        dir = os.path.join(self.base_dir, "outputs")
        return dir

    @property
    def aux_estimates_dir(self):
        dir = os.path.join(self.outputs_dir, "aux_estimates")
        if self.sub_dir is not None: dir = os.path.join(dir, self.sub_dir)
        return dir

    @property
    def observation_action_spec(self): return get_observation_action_spec_tabular(self.bounds)

    def set_dataset(self, n_pads, verbosity=0):
        hparam_str_dataset = f"{n_pads=}"
        dataset_dir = os.path.join(self.datasets_dir, hparam_str_dataset)
        self.dataset = load_or_create_dataset_Dataframe(
            dataset_dir, self.df, self.get_split, self.get_episode, self.observation_action_spec, n_pads, verbosity, )

    def set_aux_estimates(self, verbosity=0):
        assert self.dataset is not None
        self.aux_estimates = load_or_create_aux_estimates(
            self.aux_estimates_dir, self.dataset, target_policy=None, by="steps", obs_act=True, verbosity=verbosity, )

    def set_estimator(self, estimator_name):
        assert self.dataset is not None
        assert self.aux_estimates is not None
        self.estimator = self.estimators_dict[estimator_name](
            self.dataset, self.evaluation_policy, self.aux_estimates, )

    def predict(self, gamma, projected, **kwargs):
        assert self.estimator is not None
        return self.estimator.solve(gamma, projected, **kwargs)

# ---------------------------------------------------------------- #
