# ---------------------------------------------------------------- #

import tensorflow as tf

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from dice_rl.data.dataset import OffpolicyDataset
from dice_rl.utils import common as common_lib

from dice_rl.estimators.estimator import (
    _default_by_steps_reward_fn,
    _default_by_episodes_reward_fn,
    _default_by_steps_weight_fn,
    _default_by_episodes_weight_fn,
)

# ---------------------------------------------------------------- #

def get_fullbatch_average(
        dataset: OffpolicyDataset,
        limit: Optional[int] = None,
        by: str = "steps",
        truncate_episode_at: Optional[int] = None,
        reward_fn: Optional[Callable] = None,
        weight_fn: Optional[Callable] = None,
        gamma: Union[float, tf.Tensor] = 1.0,
        weighted: bool = True
    ) -> Union[float, tf.Tensor]:

    if reward_fn is None:
        if by == "steps" or by == "experience":
            reward_fn = _default_by_steps_reward_fn

        elif by == "episodes":
            reward_fn = lambda *args: \
                _default_by_episodes_reward_fn(*args, gamma=gamma)

        else:
            raise NotImplementedError

    if weight_fn is None:
        if by == "steps" or by == "experience":
            weight_fn = lambda *args: \
                _default_by_steps_weight_fn(*args, gamma=gamma)

        elif by == "episodes":
            weight_fn = _default_by_episodes_weight_fn

        else:
            raise NotImplementedError

    if by == "steps":
        steps = dataset.get_all_steps(limit=limit)
        rewards = reward_fn(steps) # type: ignore
        weights = weight_fn(steps) # type: ignore

    elif by == "episodes":
        episodes, valid_steps = dataset.get_all_episodes(
            truncate_episode_at=truncate_episode_at, limit=limit)
        rewards = reward_fn(episodes, valid_steps) # type: ignore
        weights = weight_fn(episodes, valid_steps) # type: ignore

    elif by == "experience":
        episodes, valid_steps = dataset.get_all_episodes()
        experience = tf.nest.map_structure(lambda t: t[:, 1], episodes)
        rewards = reward_fn(experience) # type: ignore
        weights = weight_fn(experience) # type: ignore

    else:
        raise NotImplementedError

    rewards = common_lib.reverse_broadcast(rewards, weights)
    weights = common_lib.reverse_broadcast(weights, rewards)

    if tf.rank(weights) < 2: # type: ignore
        a = tf.reduce_sum(rewards * weights, axis=0)
        b = tf.reduce_sum(weights, axis=0)

    else:
        a = tf.linalg.matmul(weights, rewards)
        b = tf.reduce_sum( tf.math.reduce_mean(weights, axis=0) )

    if not weighted: b = len(weights)
    return a / b

# ---------------------------------------------------------------- #
