# ---------------------------------------------------------------- #

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from dice_rl_TU_Vienna.sample import get_probs, get_probs_log
from dice_rl_TU_Vienna.estimators.tabular.utils import get_dims, obs_act_to_index

# ---------------------------------------------------------------- #

def sample_step(
        n_act,
        obs_act,
        obs_init, obs_this, act_this, rew_this, obs_next,
        step_init_target_probs, step_this_target_probs_log, step_next_target_probs,
        d0_bar, dD_bar, P_bar, r_bar, n, ):

    get_index_wrapped = lambda obs, act: obs_act_to_index(
        obs, act,
        n_act=n_act, obs_act=obs_act)

    init_probs = step_init_target_probs
    for init_action, init_prob in enumerate(init_probs):
        i_init = get_index_wrapped(obs_init, init_action)
        d0_bar[i_init] += init_prob

    i_this = get_index_wrapped(obs_this, act_this)
    dD_bar[i_this] += 1

    policy_ratio = 1.0
    if not obs_act:
        x = step_this_target_probs_log
        y = step_this.get_log_probability() # type: ignore
        policy_ratio = tf.exp(x - y)

    next_probs = step_next_target_probs
    for next_action, next_prob in enumerate(next_probs):
        i_next = get_index_wrapped(obs_next, next_action)
        P_bar[i_this, i_next] += next_prob * policy_ratio

    r_bar[i_this] += rew_this

    n += 1

    return d0_bar, dD_bar, P_bar, r_bar, n

# -------------------------------- #

def sample_by_episodes(dataset, target_policy=None, obs_act=True):

    _, n_act, dim = get_dims(dataset.spec, obs_act)

    d0_bar = np.zeros([dim])
    dD_bar = np.zeros([dim])
    P_bar  = np.zeros([dim, dim])
    r_bar  = np.zeros([dim])
    n = 0

    episodes, valid_steps = dataset.get_all_episodes(limit=None)
    num_episodes, num_steps = tf.shape(valid_steps) # type: ignore

    for i_episode in tqdm( range(num_episodes) ):

        episode_this = tf.nest.map_structure(lambda t: t[i_episode], episodes)
        step_init    = tf.nest.map_structure(lambda t: t[0], episode_this)

        step_init_target_probs = get_probs(step_init, target_policy)

        for i_step in range(num_steps - 1):
            step_this = tf.nest.map_structure(lambda t: t[i_episode, i_step], episodes)
            step_next = tf.nest.map_structure(lambda t: t[i_episode, i_step + 1], episodes)

            A = step_this.is_last() # type: ignore
            B = valid_steps[i_episode, i_step]
            if A or not B: continue

            obs_init = step_init.observation  # type: ignore
            obs_this = step_this.observation  # type: ignore
            act_this = step_this.action       # type: ignore
            rew_this = step_this.reward       # type: ignore
            obs_next = step_next.observation  # type: ignore

            step_this_target_probs_log = get_probs_log(step_this, target_policy)
            step_next_target_probs     = get_probs    (step_next, target_policy)

            d0_bar, dD_bar, P_bar, r_bar, n = sample_step(
                n_act,
                obs_act,
                obs_init, obs_this, act_this, rew_this, obs_next,
                step_init_target_probs, step_this_target_probs_log, step_next_target_probs,
                d0_bar, dD_bar, P_bar, r_bar, n, )

    return d0_bar, dD_bar, P_bar, r_bar, n


def sample_by_steps(dataset, target_policy=None, obs_act=True):

    _, n_act, dim = get_dims(dataset.spec, obs_act)

    d0_bar = np.zeros([dim])
    dD_bar = np.zeros([dim])
    P_bar  = np.zeros([dim, dim])
    r_bar  = np.zeros([dim])
    n = 0

    steps = dataset.get_all_steps(include_terminal_steps=True) # type: ignore
    num_steps = dataset.capacity # type: ignore

    for i_step in tqdm( range(num_steps-1) ): # type: ignore
        step_this = tf.nest.map_structure(lambda t: t[i_step], steps)
        step_next = tf.nest.map_structure(lambda t: t[i_step + 1], steps)

        if step_this.is_first(): # type: ignore
            step_init = step_this
            step_init_target_probs = get_probs(step_init, target_policy)

        if step_this.is_last(): continue # type: ignore

        obs_init = step_init.observation # type: ignore
        obs_this = step_this.observation # type: ignore
        act_this = step_this.action      # type: ignore
        rew_this = step_this.reward      # type: ignore
        obs_next = step_next.observation # type: ignore

        step_this_target_probs_log = get_probs_log(step_this, target_policy)
        step_next_target_probs     = get_probs    (step_next, target_policy)

        d0_bar, dD_bar, P_bar, r_bar, n = sample_step(
            n_act,
            obs_act,
            obs_init, obs_this, act_this, rew_this, obs_next,
            step_init_target_probs, step_this_target_probs_log, step_next_target_probs,
            d0_bar, dD_bar, P_bar, r_bar, n, )

    return d0_bar, dD_bar, P_bar, r_bar, n


def sample_by_experience(dataset, target_policy=None, obs_act=True):

    _, n_act, dim = get_dims(dataset.spec, obs_act)

    d0_bar = np.zeros([dim])
    dD_bar = np.zeros([dim])
    P_bar  = np.zeros([dim, dim])
    r_bar  = np.zeros([dim])
    n = 0

    experiences, valid_steps = dataset.get_all_episodes(limit=None)
    n_experience, _ = tf.shape(valid_steps) # type: ignore

    for i_experience in tqdm(range(n_experience)):

        experience = tf.nest.map_structure(lambda t: t[i_experience], experiences)
        step_init  = tf.nest.map_structure(lambda t: t[0], experience)
        step_this  = tf.nest.map_structure(lambda t: t[1], experience)
        step_next  = tf.nest.map_structure(lambda t: t[2], experience)

        obs_init = step_init.observation  # type: ignore
        obs_this = step_this.observation  # type: ignore
        act_this = step_this.action       # type: ignore
        rew_this = step_this.reward       # type: ignore
        obs_next = step_next.observation  # type: ignore

        step_init_target_probs     = get_probs    (step_init, target_policy)
        step_this_target_probs_log = get_probs_log(step_this, target_policy)
        step_next_target_probs     = get_probs    (step_next, target_policy)

        d0_bar, dD_bar, P_bar, r_bar, n = sample_step(
            n_act,
            obs_act,
            obs_init, obs_this, act_this, rew_this, obs_next,
            step_init_target_probs, step_this_target_probs_log, step_next_target_probs,
            d0_bar, dD_bar, P_bar, r_bar, n, )

    return d0_bar, dD_bar, P_bar, r_bar, n

# -------------------------------- #

samplers = {
    "episodes":   sample_by_episodes,
    "steps":      sample_by_steps,
    "experience": sample_by_experience,
}

def sample(by, dataset, target_policy=None, obs_act=True):
    return samplers[by](dataset, target_policy, obs_act)

# ---------------------------------------------------------------- #
