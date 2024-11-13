# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

import os

from tqdm import tqdm

from utils.numpy import add_middle_means_log

# ---------------------------------------------------------------- #

def get_gammas_1():
    return np.arange(0.1, 1, 0.1)

def get_gammas_2(n=5):
    one_minus_gammas = add_middle_means_log(
        0.1 ** np.arange(1, n+1), 2)
    gammas = 1 - one_minus_gammas

    return gammas

# ---------------------------------------------------------------- #

def get_pv_OnPE(estimator, gamma, *args):
    pv = estimator(gamma)
    return pv

def get_pv_OffPE(estimator, gamma, *args):
    projected, weighted, modified, lam = args

    pv, *_ = estimator.solve(
        gamma, projected,
        weighted=weighted,
        modified=modified, lam=lam,
    )
    return pv

def get_sdc(estimator, gamma, *args):
    projected, modified, lam = args

    sdc, _ = estimator.solve_sdc(
        gamma, projected,
        modified=modified, lam=lam,
    )
    return sdc

def get_vaf(estimator, gamma, *args):
    projected, = args

    vaf, _ = estimator.solve_vaf(
        gamma, projected,
    )
    return vaf


def apply_get(
        get, hparam_str, file_name,
        estimator_s, gamma_s,
        save_dir=None, verbosity=0,
        *args):

    if isinstance(estimator_s, list):
        pbar = estimator_s
        if verbosity == 1: pbar = tqdm(pbar)
        return np.array([
            apply_get(
                get, hparam_str, file_name,
                estimator, gamma_s,
                save_dir, verbosity,
                *args
            )
                for estimator in pbar
        ])

    if isinstance(gamma_s, list) or isinstance(gamma_s, np.ndarray):
        try:
            if save_dir is not None:
                save_dir = os.path.join(save_dir, estimator_s.__name__)

                hs = hparam_str.get(estimator_s.__name__, None)
                if hs is not None: save_dir = os.path.join(save_dir, hs)

                if not tf.io.gfile.isdir(save_dir):
                    tf.io.gfile.makedirs(save_dir)

                path = os.path.join(save_dir, f"{file_name}.npy")

                if verbosity == 2: print(f"Try loading {path}")
                array = np.load(path)

            else: raise FileNotFoundError

        except:
            if verbosity == 2: print(f"No {file_name} found in", path)
            pbar = gamma_s
            if verbosity == 2: pbar = tqdm(pbar)
            array = np.array([
                apply_get(
                    get, hparam_str, file_name,
                    estimator_s, gamma,
                    save_dir, verbosity,
                    *args
                )
                    for gamma in pbar
            ])

            if save_dir is not None:
                np.save(path, array)
                if verbosity == 2: print(f"saved {path}")

        return array

    return get(estimator_s, gamma_s, *args)


def get_pv_s_OnPE(
        estimator_s, gamma_s,
        save_dir=None, verbosity=0):

    hparam_str = {}

    return apply_get(
        get_pv_OnPE, hparam_str, "pv",
        estimator_s, gamma_s,
        save_dir, verbosity)

def get_pv_s_OffPE(
        estimator_s, gamma_s,
        projected, weighted, modified, lam,
        save_dir=None, verbosity=0):

    hparam_str = {
        "TabularVafe": f"{projected=}",
        "TabularDice": f"{projected=}_{weighted=}_{modified=}",
        "TabularDualDice": f"{projected=}_{weighted=}",
        "TabularGradientDice": f"{projected=}_{weighted=}_{lam=}",
    }

    return apply_get(
        get_pv_OffPE, hparam_str, "pv",
        estimator_s, gamma_s,
        save_dir, verbosity,
        projected, weighted, modified, lam)

def get_sdc_s(
        estimator_s, gamma_s,
        projected, modified, lam,
        save_dir=None, verbosity=0):

    hparam_str = {
        "TabularDice": f"{projected=}_{modified=}",
        "TabularDualDice": f"{projected=}",
        "TabularGradientDice": f"{projected=}_{lam=}",
    }

    return apply_get(
        get_sdc, hparam_str, "sdc",
        estimator_s, gamma_s,
        save_dir, verbosity,
        projected, modified, lam)

def get_vaf_s(
        estimator_s, gamma_s,
        projected,
        save_dir=None, verbosity=0):

    hparam_str = {
        "TabularVafe": f"{projected=}",
    }

    return apply_get(
        get_vaf, hparam_str, "vaf",
        estimator_s, gamma_s,
        save_dir, verbosity,
        projected)

# ---------------------------------------------------------------- #
