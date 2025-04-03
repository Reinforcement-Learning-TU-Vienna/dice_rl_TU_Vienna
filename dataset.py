# ---------------------------------------------------------------- #

import os

import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime

from dice_rl_TU_Vienna.utils.seeds import set_all_seeds
from dice_rl_TU_Vienna.utils.json import (
    json_get_id, json_remove_by_dict, json_append, )

# ---------------------------------------------------------------- #

def create_dataset_from_env(env, get_act, hyperparameters, verbosity=0):
    set_all_seeds(hyperparameters["seed"])

    n_samples = hyperparameters.get("n_samples", None)
    n_trajectories = hyperparameters.get("n_trajectories", None)

    dataset = {
        "obs_init": [],
        "obs": [],
        "act": [],
        "rew": [],
        "obs_next": [],
    }

    by_samples = n_samples is not None
    by_trajectories = n_trajectories is not None

    if by_samples:
        assert not by_trajectories

        pbar = range(n_samples)
        if verbosity > 0: pbar = tqdm(pbar)

        for _ in pbar:

            obs_init, _ = env.reset()

            obs = env.observation_space.sample()
            env.unwrapped.s = obs
            act = get_act(obs)
            obs_next, rew, _, _, _ = env.step(act)

            K = ["obs_init", "obs", "act", "rew", "obs_next"]
            V = [obs_init, obs, act, rew, obs_next]
            for k, v in zip(K, V):
                dataset[k].append(v)

    if by_trajectories:
        assert not by_samples

        dataset["id"] = []
        dataset["t"] = []

        pbar = range(n_trajectories)
        if verbosity > 0: pbar = tqdm(pbar)

        for id in pbar:
            t = 0
            obs_init, _ = env.reset()
            obs = obs_init

            done = False
            while not done:
                act = get_act(obs)
                obs_next, rew, terminated, truncated, _ = env.step(act)

                K = ["id", "t", "obs_init", "obs", "act", "rew", "obs_next"]
                V = [id, t, obs_init, obs, act, rew, obs_next]
                for k, v in zip(K, V):
                    dataset[k].append(v)

                t += 1
                obs = obs_next
                done = terminated or truncated

    return pd.DataFrame(dataset)


def save_dataset(dir, dataset, hyperparameters, verbosity=0):
    file_path_json = os.path.join(dir, "dataset.json")

    id_dataset = datetime.now().isoformat()
    dir_dataset = os.path.join(dir, id_dataset)
    file_path_parquet = os.path.join(dir_dataset, "dataset.parquet")

    hyperparameters_labeled = {
        "id": id_dataset,
        "data": hyperparameters,
    }

    json_append(file_path=file_path_json, dictionary=hyperparameters_labeled)

    if not tf.io.gfile.isdir(dir_dataset):
        tf.io.gfile.makedirs(dir_dataset)

    dataset.to_parquet(file_path_parquet)
    if verbosity > 0: print(f"saved {file_path_parquet}")

    return id_dataset


def get_dataset(
        dir,
        create_dataset, args_create_dataset,
        hyperparameters,
        postprocessing=None,
        verbosity=0,
    ):

    loaded = False
    id_dataset = None
    if dir is not None:
        file_path_json = os.path.join(dir, "dataset.json")
        if verbosity > 0: print(f"trying to find id_dataset in {file_path_json}")
        id_dataset = json_get_id(file_path=file_path_json, dictionary=hyperparameters)

    if id_dataset is not None:
        dir_dataset = os.path.join(dir, id_dataset)
        file_path_parquet = os.path.join(dir_dataset, f"dataset.parquet")

        if verbosity > 0: print(f"trying to load dataset from {file_path_parquet}")
        if os.path.exists(file_path_parquet):
            dataset = pd.read_parquet(file_path_parquet)
            loaded = True

    if not loaded:
        if verbosity > 0: print("failed to load")
        if verbosity > 0: print("creating dataset from environment")

        dataset = create_dataset(*args_create_dataset, hyperparameters, verbosity)
        if postprocessing is not None:
            dataset = postprocessing(dataset)

        if dir is not None:
            if verbosity > 0: print(f"removing dataset hyperparameters from {file_path_json}")
            json_remove_by_dict(file_path=file_path_json, dictionary=hyperparameters)

            if verbosity > 0: print("saving dataset")
            id_dataset = save_dataset(dir, dataset, hyperparameters, verbosity)

    if dir is not None: assert id_dataset is not None
    return dataset, id_dataset


def get_dataset_from_env(
        dir,
        env, get_act,
        hyperparameters,
        postprocessing=None,
        verbosity=0,
    ):

    create_dataset = create_dataset_from_env
    args_create_dataset = (env, get_act)
    return get_dataset(
        dir,
        create_dataset, args_create_dataset,
        hyperparameters,
        postprocessing,
        verbosity,
    )


def get_dataset_from_df(
        dir,
        df,
        hyperparameters,
        postprocessing=None,
        verbosity=0,
    ):

    create_dataset = lambda hyperparameters, verbosity=0: df
    args_create_dataset = ()
    return get_dataset(
        dir,
        create_dataset, args_create_dataset,
        hyperparameters,
        postprocessing,
        verbosity,
    )

# ---------------------------------------------------------------- #
