# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from gymnasium.wrappers.time_limit import TimeLimit

from tf_agents.environments import TFEnvironment
from tf_agents.trajectories.time_step import TimeStep, time_step_spec

# ---------------------------------------------------------------- #

class MyTFEnvironment(TFEnvironment):
    def __init__(self, env, observation_spec, action_spec, reward_spec=None):
        self.env = env
        self._time_step_spec = time_step_spec(observation_spec, reward_spec)
        self._action_spec = action_spec

    def _reset(self):
        observation, info = self.env.reset()

        time_step = get_time_step(
            [0], 0, 1, observation,
            self.time_step_spec() )

        self._current_time_step_stored = time_step
        return time_step

    def _step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        time_step = get_time_step(
            [2 if terminated else 1], reward, 1, observation,
            self.time_step_spec() )

        self._current_time_step_stored = time_step
        return time_step

    def _current_time_step(self):
        return self._current_time_step_stored

# -------------------------------- #

def get_time_step(
        step_type, reward, discount, observation,
        time_step_spec):

    v_1 = [step_type]
    v_2 = reward
    v_3 = discount
    v_4 = observation

    dtype_1 = time_step_spec.step_type.dtype
    dtype_2 = time_step_spec.reward.dtype
    dtype_3 = time_step_spec.discount.dtype
    dtype_4 = time_step_spec.observation.dtype

    step_type   = tf.convert_to_tensor(v_1, dtype=dtype_1, name="step_type")
    reward      = tf.convert_to_tensor(v_2, dtype=dtype_2, name="reward")
    discount    = tf.convert_to_tensor(v_3, dtype=dtype_3, name="discount")
    observation = tf.convert_to_tensor(v_4, dtype=dtype_4, name="observation")

    return TimeStep(step_type, reward, discount, observation)

# ---------------------------------------------------------------- #

def test_env(
        env, get_act,
        num_trajectory, max_trajectory_length=None,
        verbosity=1):

    if max_trajectory_length is not None:
        env = TimeLimit(env, max_episode_steps=max_trajectory_length)

    num_obs_per_episode = {
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

        num_obs_per_episode["distinct_absolute"].append( len(obs_set) )
        num_obs_per_episode["distinct_relative"].append( len(obs_set) / (counter+1) )
        num_obs_per_episode["all"].append( counter+1 )

    return {
        k: ( np.mean(v), np.std(v) )
            for k, v in num_obs_per_episode.items()
    }

# ---------------------------------------------------------------- #

