# ---------------------------------------------------------------- #

import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime

from dice_rl_TU_Vienna.estimators.tabular.tabular_OffPE import TabularOffPE
from dice_rl_TU_Vienna.utils.seeds import set_all_seeds
from dice_rl_TU_Vienna.utils.json import json_append

# ---------------------------------------------------------------- #

class TabularSarsa(TabularOffPE):
    @property
    def __name__(self): return "TabularSarsa"

    def get_value(self, vector, obs, act):
        index = self.indexer.get_index(obs, act)
        value = vector[index]

        return value

    def get_average_value(self, vector, obs, probs):
        scalar_obs = np.ndim(obs) == 0
        if scalar_obs: obs = np.array([obs])

        obs_all = np.repeat(obs, self.n_act)
        act_all = np.concatenate([ np.arange(self.n_act) ] * len(obs))

        values_all = self.get_value(vector, obs_all, act_all)
        values = np.reshape(values_all, [len(obs), self.n_act])

        average_value = np.sum(values * probs, axis=1)
        if scalar_obs: average_value = average_value[0]

        return average_value

    def update_Q_hat_by_sample(self, Q_hat, sample, gamma, alpha):
        obs = sample["obs"]
        act = sample["act"]
        rew = sample["rew"]
        obs_next   = sample["obs_next"]
        probs_next = sample["probs_next"]

        Q = self.get_value(Q_hat, obs, act)
        Q_next = self.get_average_value(Q_hat, obs_next, probs_next)

        bellman_error = rew + gamma * Q_next - Q
        index = self.get_index(obs, act)
        Q_hat[index] += alpha * bellman_error

        return Q_hat

    def update_Q_hat_by_episode(self, Q_hat, episode, gamma, alpha):
        indices = np.array(episode.index)

        for i in indices[::-1]:
            sample = episode.loc[i]
            Q_hat = self.update_Q_hat_by_sample(Q_hat, sample, gamma, alpha)

        return Q_hat

    def solve_vaf_n_steps(
            self, gamma, n_steps, by, alpha, get_metrics, verbosity, Q_hat, info):

        pbar = range(n_steps)
        if verbosity >= 1: pbar = tqdm(pbar)

        for i in pbar:
            rho_hat = self.solve_pv(gamma, Q_hat)
            metrics = get_metrics(Q_hat, rho_hat)
            for k, v in metrics.items():
                info["metrics"][k].append(v)
            if verbosity >= 1: pbar.set_postfix(metrics) # type: ignore

            if by == "samples":
                sample = self.dataset.sample()
                Q_hat = self.update_Q_hat_by_sample(Q_hat, sample, gamma, alpha)

            if by == "episodes":
                id = self.dataset["id"].sample().iloc[0]
                f = self.dataset["id"] == id
                episode = self.dataset[f]
                Q_hat = self.update_Q_hat_by_episode(Q_hat, episode, gamma, alpha)

        return Q_hat, info

    def solve_vaf_n_epochs(
            self, gamma, n_epochs, by, alpha, get_metrics, shuffle, verbosity, Q_hat, info):

        pbar = range(n_epochs)
        if verbosity == 1: pbar = tqdm(pbar)

        for j in pbar:
            rho_hat = self.solve_pv(gamma, Q_hat)
            metrics = get_metrics(Q_hat, rho_hat)
            if verbosity == 1: pbar.set_postfix(metrics) # type: ignore

            if verbosity >= 2: print(f"epoch {j+1}/{n_epochs}")

            if by == "samples":
                indices = np.array(self.dataset.index)
                if shuffle: np.random.shuffle(indices)

                pbar = indices
                if verbosity >= 2: pbar = tqdm(pbar)

                for i in indices:
                    rho_hat = self.solve_pv(gamma, Q_hat)
                    metrics = get_metrics(Q_hat, rho_hat)
                    for k, v in metrics.items():
                        info["metrics"][k].append(v)
                    if verbosity >= 2: pbar.set_postfix(metrics) # type: ignore

                    sample = self.dataset.loc[i]
                    Q_hat = self.update_Q_hat_by_sample(Q_hat, sample, gamma, alpha)

            if by == "episodes":
                ids = np.array( self.dataset["id"].unique() )
                if shuffle: np.random.shuffle(ids)

                pbar = ids
                if verbosity >= 2: pbar = tqdm(pbar)

                for id in pbar:
                    rho_hat = self.solve_pv(gamma, Q_hat)
                    metrics = get_metrics(Q_hat, rho_hat)
                    for k, v in metrics.items():
                        info["metrics"][k].append(v)
                    if verbosity >= 2: pbar.set_postfix(metrics) # type: ignore

                    f = self.dataset["id"] == id
                    episode = self.dataset[f]
                    Q_hat = self.update_Q_hat_by_episode(Q_hat, episode, gamma, alpha)

        return Q_hat, info

    def solve_vaf(self, gamma, **kwargs):
        n_steps  = kwargs.get("n_steps",  None)
        n_epochs = kwargs.get("n_epochs", None)

        by = kwargs["by"]

        alpha   = kwargs["alpha"]
        shuffle = kwargs.get("shuffle", False)

        verbosity = kwargs.get("verbosity", 0)
        get_metrics = kwargs.get("get_metrics", lambda Q_hat, rho_hat: {})

        Q_hat = np.zeros(self.dimension)
        rho_hat = self.solve_pv(gamma, Q_hat)
        info = { "metrics": { key: [] for key in get_metrics(Q_hat, rho_hat) } }

        if n_steps is not None:
            assert n_epochs is None
            Q_hat, info = self.solve_vaf_n_steps(
                gamma, n_steps, by, alpha, get_metrics, verbosity, Q_hat, info)

        if n_epochs is not None:
            assert n_steps is None
            Q_hat, info = self.solve_vaf_n_epochs(
                gamma, n_epochs, by, alpha, get_metrics, shuffle, verbosity, Q_hat, info)

        return Q_hat, info

    def solve_pv(self, gamma, Q_hat):
        dataset_init = self.dataset
        if "t" in self.dataset.columns:
            f = self.dataset["t"] == 0
            dataset_init = self.dataset[f]

        obs   = np.array(dataset_init["obs_init"])
        probs = np.stack(dataset_init["probs_init"].values)

        R_hat = np.mean( self.get_average_value(Q_hat, obs, probs) )
        rho_hat = (1 - gamma) * R_hat

        return rho_hat

    def solve(self, gamma, **kwargs):
        """
        Args:
            gamma (float): discount factor.
            seed (int, optional): Set all seeds before executing method. Defaults to `None`.
            n_steps (int, optional): number of steps. Defaults to `None`.
            n_epochs (int, optional): number of epochs. Defaults to `None`.
            by (str): `"samples"` or `"episodes"`. The latter requires an `"id"`-column in the `dataset`.
            alpha (float): learning rate.
            shuffle (bool): Shuffle dataset before each epoch. Defaults to `False`.
            verbosity (int): level of detail in output or messages. Defaults to `0`.
            get_metrics (callable): function with args `(Q_hat, rho_hat)` and returns `dict` every step. Defaults to `None`.
            dir_save (str): path where metrics get saved after completion. Defaults to `None`.
        """
        seed = kwargs.get("seed", None)
        set_all_seeds(seed)

        Q_hat, info = self.solve_vaf(gamma, **kwargs)
        rho_hat = self.solve_pv(gamma, Q_hat)

        info["Q_hat"] = Q_hat # type: ignore

        if ( dir_save := kwargs.get("dir_save", None) ) is not None:
            id = datetime.now().isoformat()

            hyperparameters = {
                "name": "TabularSarsa",
                "gamma": gamma,
                "seed": seed,
                "n_steps": kwargs.get("n_steps", None),
                "n_epochs": kwargs.get("n_epochs", None),
                "by": kwargs["by"],
                "alpha": kwargs["alpha"],
                "shuffle": kwargs.get("shuffle", False),
            }

            json_append(
                file_path=os.path.join(dir_save, "evaluation.json"),
                dictionary=hyperparameters,
                id=id,
            )

            path = os.path.join(dir_save, id)

            for k, v in info["metrics"].items():

                if not tf.io.gfile.isdir(path):
                    tf.io.gfile.makedirs(path)

                file = os.path.join(path, f"{k}.npy")
                arr = np.array(v)
                np.save(file, arr)

        return rho_hat, info

# ---------------------------------------------------------------- #
