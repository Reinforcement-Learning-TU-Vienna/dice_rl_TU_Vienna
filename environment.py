# ---------------------------------------------------------------- #

import numpy as np

from tqdm import tqdm

from gymnasium.wrappers import TimeLimit

# ---------------------------------------------------------------- #

def test_env(
        env, get_act,
        num_trajectory, max_trajectory_length=None,
        verbosity=1):

    if max_trajectory_length is not None:
        env = TimeLimit(env, max_episode_steps=max_trajectory_length)

    n_obs_per_episode = {
        "distinct_absolute": [],
        "distinct_relative": [],
        "all": [],
    }

    pbar = range(num_trajectory)
    if verbosity == 1: pbar = tqdm(pbar)

    for _ in pbar:
        obs_set = set()
        terminated = False
        truncated = False
        counter = 0

        obs, _ = env.reset()
        if verbosity == 2: print(obs, end=", ")
        if verbosity >= 3: print(obs)
        obs_set.add(obs)

        while not (terminated or truncated):
            counter += 1

            act = get_act(obs)
            obs, rew, terminated, truncated, info = env.step(act)

            if verbosity == 2: print(act, rew); print(obs, end=", ")
            if verbosity >= 3: print(act, "->", obs, rew, terminated, truncated, info)

            obs_set.add(obs)

        if verbosity == 2:
            if terminated: print("terminated")
            if truncated:  print("truncated")
        if verbosity >= 2: print()

        n_obs_per_episode["distinct_absolute"].append( len(obs_set) )
        n_obs_per_episode["distinct_relative"].append( len(obs_set) / (counter+1) )
        n_obs_per_episode["all"].append( counter+1 )

    return {
        k: { "mean": np.mean(v), "std": np.std(v), }
            for k, v in n_obs_per_episode.items()
    }

# ---------------------------------------------------------------- #
