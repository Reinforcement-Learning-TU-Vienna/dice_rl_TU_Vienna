# ---------------------------------------------------------------- #

import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from dice_rl_TU_Vienna.utils.numpy import safe_divide
from dice_rl_TU_Vienna.estimators.tabular.tabular_OffPE import Indexer, TabularOffPE

# ---------------------------------------------------------------- #

class AuxiliaryEstimates(object):
    names = ["d0_bar", "dD_bar", "P_bar", "r_bar", "n", ]

    def __init__(self, dataset, n_obs, n_act, path=None, verbosity=0):
        self.dataset = dataset
        self.n_obs = n_obs
        self.n_act = n_act
        self.path = path
        self.verbosity = verbosity

        self.indexer = Indexer(n_obs, n_act)

        try:
            if verbosity > 0: print(f"trying to load auxiliary estimates from {self.path}")
            self.load()
        except:
            if verbosity > 0: print("failed to load")
            if verbosity > 0: print("creating auxiliary estimates from dataset")
            self.create()
            if verbosity > 0: print("saving auxiliary estimates")
            self.save()

    def load(self):
        assert self.path is not None

        E = []
        for name in self.names:
            file_path = os.path.join(self.path, f"{name}.npy")
            e = np.load(file_path)
            if self.verbosity > 0: print(f"loaded {file_path}")
            E.append(e)

        self.d0_bar = E[0]
        self.dD_bar = E[1]
        self.P_bar  = E[2]
        self.r_bar  = E[3]
        self.n      = E[4]

    def save(self):
        if self.path is None: return

        E = [
            self.d0_bar,
            self.dD_bar,
            self.P_bar,
            self.r_bar,
            self.n,
        ]

        if not tf.io.gfile.isdir(self.path):
            tf.io.gfile.makedirs(self.path)

        for name, e in zip(self.names, E):
            file_path = os.path.join(self.path, f"{name}.npy")
            np.save(file_path, e)
            if self.verbosity > 0: print(f"saved {file_path}")

    def create(self):
        self.d0_bar = np.zeros(self.indexer.dimension)
        self.dD_bar = np.zeros(self.indexer.dimension)
        self.P_bar  = np.zeros([self.indexer.dimension]*2)
        self.r_bar  = np.zeros(self.indexer.dimension)
        self.n      = len(self.dataset)

        pbar = self.dataset.iterrows()
        if self.verbosity > 0: pbar = tqdm(pbar, total=self.n)

        for _, experience in pbar:

            for act_init, prob_init in enumerate(experience.probs_init):
                index_init = self.indexer.get_index(experience.obs_init, act_init)
                self.d0_bar[index_init] += prob_init

            index = self.indexer.get_index(experience.obs, experience.act)
            self.dD_bar[index] += 1

            for act_next, prob_next in enumerate(experience.probs_next):
                index_next = self.indexer.get_index(experience.obs_next, act_next)
                self.P_bar[index, index_next] += prob_next

            self.r_bar[index] += experience.rew

    @property
    def bar(self): return self.d0_bar, self.dD_bar, self.P_bar, self.r_bar, self.n

    @property
    def hat(self):
        d0_hat = self.d0_bar / self.n
        dD_hat = self.dD_bar / self.n
        P_hat = safe_divide(self.P_bar.T, self.dD_bar).T
        r_hat = safe_divide(self.r_bar, self.dD_bar)
        return d0_hat, dD_hat, P_hat, r_hat

# ---------------------------------------------------------------- #

class ModelBasedTabularOffPE(TabularOffPE):

    def __init__(self, dataset, n_obs, n_act, path=None, verbosity=0, auxiliary_estimates=None):
        super().__init__(dataset, n_obs, n_act)

        if auxiliary_estimates is None:
            auxiliary_estimates = AuxiliaryEstimates(dataset, n_obs, n_act, path, verbosity)

        self.auxiliary_estimates = auxiliary_estimates

# ---------------------------------------------------------------- #
