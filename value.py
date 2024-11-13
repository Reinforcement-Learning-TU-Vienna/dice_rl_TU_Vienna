# ---------------------------------------------------------------- #

import numpy as np

from tqdm import tqdm

from gymnasium.wrappers.time_limit import TimeLimit

from dice_rl_TU_Vienna.dataset import get_all_episodes

from utils.numpy import pad

# ---------------------------------------------------------------- #

def get_rewards_matrix_env(
        env, get_act,
        num_trajectory, pad_rew, max_trajectory_length=None,
        verbosity=0):

    if max_trajectory_length is not None:
        env = TimeLimit(env, max_episode_steps=max_trajectory_length)

    rewards = [ [] for _ in range(num_trajectory) ]

    pbar = range(num_trajectory)
    if verbosity == 1: pbar = tqdm(pbar)
    for s in pbar:
        obs, _ = env.reset()

        terminated = False
        truncated = False
        while not (terminated or truncated):
            act = get_act(obs)
            obs, rew, terminated, truncated, _ = env.step(act)

            rewards[s].append(rew)

    rewards, max_trajectory_length = pad(rewards, pad_rew, verbosity=verbosity)

    return rewards, max_trajectory_length


def get_rewards_matrix_dataset(dataset, by="steps", verbosity=0):
    episodes = get_all_episodes(dataset, by, verbosity=verbosity)
    rewards = [episode.reward for episode in episodes] # type: ignore

    rewards, max_trajectory_length = pad(rewards)

    return rewards, max_trajectory_length

# ---------------------------------------------------------------- #

def get_get_policy_value(rewards, max_trajectory_length):
    def get_policy_value(gamma, scale=False):

        H = max_trajectory_length

        discounts = gamma ** np.arange(H)
        returns = rewards @ discounts

        v = np.mean(returns)

        if not scale:
            factor =  1 - gamma if gamma < 1 else 1 / H
            v *= factor

        return v

    return get_policy_value


def get_get_policy_value_env(env, get_act, num_trajectory, pad_rew=0, verbosity=0):

    rewards, max_trajectory_length = get_rewards_matrix_env(
        env, get_act, num_trajectory, pad_rew, verbosity=verbosity)

    get_policy_value = get_get_policy_value(
        rewards, max_trajectory_length)

    return get_policy_value, rewards


def get_get_policy_value_dataset(dataset, by="steps", verbosity=0):

    rewards, max_trajectory_length = get_rewards_matrix_dataset(
        dataset, by, verbosity=verbosity)

    get_policy_value = get_get_policy_value(
        rewards, max_trajectory_length)

    return get_policy_value, rewards

# -------------------------------- #

def get_success_rate_env(env, get_act, num_trajectory, verbosity=0):
    returns = np.zeros(num_trajectory)

    pbar = range(num_trajectory)
    if verbosity == 1: pbar = tqdm(pbar)
    for i in pbar:
        obs, _ = env.reset()

        terminated = False
        truncated = False
        while not (terminated or truncated):
            act = get_act(obs)
            obs, rew, terminated, truncated, _ = env.step(act)

            returns[i] += rew

    return np.cumsum(returns) / ( np.arange( len(returns) ) + 1)


def get_success_rate_dataset(dataset, by="steps"):
    returns = [
        np.sum(episode.reward) # type: ignore
            for episode in get_all_episodes(dataset, by=by)
    ]

    return np.cumsum(returns) / ( np.arange( len(returns) ) + 1 )

# ---------------------------------------------------------------- #
