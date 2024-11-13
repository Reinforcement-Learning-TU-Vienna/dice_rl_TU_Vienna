import tensorflow as tf
import numpy as np

from tqdm import tqdm

from dice_rl.data.dataset import convert_to_tfagents_timestep


def monte_carlo(dataset, target_policy, gamma=0.99):
    episodes, valid_steps = dataset.get_all_episodes()
    num_episodes, _ = tf.shape(valid_steps) # type: ignore

    returns = []

    for episode_num in tqdm( range(num_episodes) ):

        episode = tf.nest.map_structure(
            lambda t: t[episode_num, :-1], episodes)
        this_tfagents_episode = convert_to_tfagents_timestep(episode) # type: ignore

        episode_target_log_probabilities = target_policy \
            .distribution(this_tfagents_episode) \
            .action \
            .log_prob(episode.action) # type: ignore

        x = episode_target_log_probabilities
        y = episode.get_log_probability() # type: ignore
        policy_ratios = tf.exp(x - y)
        Pi = tf.reduce_prod(policy_ratios)

        r = episode.reward # type: ignore
        if gamma < 1:
            t = tf.cast(episode.step_num, dtype=tf.float32) # type: ignore
            R = (1 - gamma) * tf.reduce_sum(gamma ** t *  r) # type: ignore
        else:
            R = tf.reduce_mean(r)

        returns.append(Pi * R)

    return np.average(returns)