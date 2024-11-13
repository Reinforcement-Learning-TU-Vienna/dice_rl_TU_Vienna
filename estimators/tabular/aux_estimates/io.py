# ---------------------------------------------------------------- #

import os

import tensorflow as tf
import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.aux_estimates.sample import sample

# ---------------------------------------------------------------- #

aux_estimates_file_names = [
    "d0_bar", "dD_bar", "P_bar", "r_bar", "n", ]

# ---------------------------------------------------------------- #

def save_aux_estimates(
        aux_estimates, aux_estimates_dir,
        by, obs_act=True,
        verbosity=0):

    hparam_str_aux_estimate = f"by={by}_{obs_act=}"
    save_dir = os.path.join(aux_estimates_dir, hparam_str_aux_estimate)

    if not tf.io.gfile.isdir(save_dir):
        tf.io.gfile.makedirs(save_dir)

    for array, file_name in zip(aux_estimates, aux_estimates_file_names):
        path = os.path.join(save_dir, file_name)
        np.save(path, array)
        if verbosity == 1: print(f"saved {path}")


def load_aux_estimates(
        aux_estimates_dir,
        by, obs_act=True,
        verbosity=0):

    hparam_str_aux_estimate = f"by={by}_{obs_act=}"
    load_dir = os.path.join(aux_estimates_dir, hparam_str_aux_estimate)

    aux_estimates = []

    for file_name in aux_estimates_file_names:
        path = os.path.join(load_dir, f"{file_name}.npy")
        aux_estimates.append( np.load(path) )
        if verbosity == 1: print(f"loaded {path}")

    return tuple(aux_estimates)


def load_or_create_aux_estimates(
        aux_estimates_dir,
        dataset, target_policy,
        by, obs_act=True,
        verbosity=0):

    try:
        aux_estimates = load_aux_estimates(
            aux_estimates_dir,
            by, obs_act,
            verbosity)

    except KeyboardInterrupt:
        if verbosity == 1: print("KeyboardInterrupt")
        assert False

    except:
        aux_estimates = sample(
            by,
            dataset, target_policy,
            obs_act)

        save_aux_estimates(
            aux_estimates,
            aux_estimates_dir,
            by, obs_act,
            verbosity)

    return aux_estimates

# ---------------------------------------------------------------- #
