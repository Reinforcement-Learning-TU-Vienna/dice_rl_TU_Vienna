# ---------------------------------------------------------------- #

import os

import tensorflow as tf
import numpy as np

from dice_rl_TU_Vienna.estimators.tabular.auxiliary_estimates.sample import sample

# ---------------------------------------------------------------- #

auxiliary_estimates_file_names = [
    "d0_bar", "dD_bar", "P_bar", "r_bar", "n", ]

# ---------------------------------------------------------------- #

def save_auxiliary_estimates(
        auxiliary_estimates, auxiliary_estimates_dir,
        by, obs_act=True,
        verbosity=0):

    hparam_str_aux_estimate = f"by={by}_{obs_act=}"
    save_dir = os.path.join(auxiliary_estimates_dir, hparam_str_aux_estimate)

    if not tf.io.gfile.isdir(save_dir):
        tf.io.gfile.makedirs(save_dir)

    for array, file_name in zip(auxiliary_estimates, auxiliary_estimates_file_names):
        path = os.path.join(save_dir, file_name)
        np.save(path, array)
        if verbosity == 1: print(f"saved {path}")


def load_auxiliary_estimates(
        auxiliary_estimates_dir,
        by, obs_act=True,
        verbosity=0):

    hparam_str_aux_estimate = f"by={by}_{obs_act=}"
    load_dir = os.path.join(auxiliary_estimates_dir, hparam_str_aux_estimate)

    auxiliary_estimates = []

    for file_name in auxiliary_estimates_file_names:
        path = os.path.join(load_dir, f"{file_name}.npy")
        auxiliary_estimates.append( np.load(path) )
        if verbosity == 1: print(f"loaded {path}")

    return tuple(auxiliary_estimates)


def load_or_create_auxiliary_estimates(
        auxiliary_estimates_dir,
        dataset, target_policy=None,
        by="steps", obs_act=True,
        verbosity=0):

    try:
        auxiliary_estimates = load_auxiliary_estimates(
            auxiliary_estimates_dir,
            by, obs_act,
            verbosity)

    except KeyboardInterrupt:
        if verbosity == 1: print("KeyboardInterrupt")
        assert False

    except:
        auxiliary_estimates = sample(by, dataset, target_policy, obs_act)

        save_auxiliary_estimates(
            auxiliary_estimates,
            auxiliary_estimates_dir,
            by, obs_act,
            verbosity)

    return auxiliary_estimates

# ---------------------------------------------------------------- #
