# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf


from abc import ABC, abstractmethod

# ---------------------------------------------------------------- #

def obs_act_to_index(
        obs, act,
        n_obs=None, n_act=None,
        neighbours="act"):

    o = tf.cast(obs, dtype=tf.int64)
    a = tf.cast(act, dtype=tf.int64)

    if neighbours == "act":
        n_a = tf.cast(n_act, dtype=tf.int64)
        return o * n_a + a # type: ignore

    if neighbours == "obs":
        n_o = tf.cast(n_obs, dtype=tf.int64)
        return o + n_o * a # type: ignore

    raise ValueError

def index_to_obs_act(
        index,
        n_obs=None, n_act=None,
        neighbours="act"):

    n_a = tf.cast(n_act, dtype=tf.int64)
    n_o = tf.cast(n_obs, dtype=tf.int64)

    if neighbours == "act": return index // n_a, index %  n_a
    if neighbours == "obs": return index %  n_o, index // n_o
    raise ValueError


class Indexer:
    def __init__(self, n_obs, n_act):
        self.n_obs = n_obs
        self.n_act = n_act

    @property
    def dimension(self): return self.n_obs * self.n_act

    def get_index(self, obs, act):
        return obs_act_to_index(
            obs, act,
            self.n_obs, self.n_act,
            neighbours="act", )


class TabularOffPE(ABC):
    @property
    @abstractmethod
    def __name__(self): pass

    def __init__(self, dataset, n_obs, n_act):
        self.dataset = dataset
        self.indexer = Indexer(n_obs, n_act)

    @property
    def n_obs(self): return self.indexer.n_obs

    @property
    def n_act(self): return self.indexer.n_act

    @property
    def dimension(self): return self.indexer.dimension

    def get_index(self, obs, act): return self.indexer.get_index(obs, act)

    @abstractmethod
    def solve(self, gamma, **kwargs):
        pass

# ---------------------------------------------------------------- #
