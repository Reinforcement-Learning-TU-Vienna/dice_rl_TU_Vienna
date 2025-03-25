# ---------------------------------------------------------------- #

import numpy as np

from tqdm import tqdm

from dice_rl_TU_Vienna.utils.numpy import add_middle_means_log

# ---------------------------------------------------------------- #

def get_gammas(min=0.1, max=0.9, step=0.1):
    return np.arange(min, max+step, step)

def get_gammas_log10(e_min=1, e_max=5, refinement=2):
    one_minus_gammas = add_middle_means_log(
        0.1 ** np.arange(e_min, e_max+1), refinement)
    gammas = 1 - one_minus_gammas

    return gammas

# ---------------------------------------------------------------- #

def get_pv(estimator, gamma, **kwargs):
    pv, _ = estimator.solve(gamma, **kwargs)
    return pv

def get_sdc(estimator, gamma, **kwargs):
    sdc, _ = estimator.solve_sdc(gamma, **kwargs)
    return sdc

def get_vaf(estimator, gamma, **kwargs):
    vaf, _ = estimator.solve_vaf(gamma, **kwargs)
    return vaf


def apply_get(
        get,
        estimator_s, gamma_s,
        verbosity=0,
        **kwargs):

    if isinstance(estimator_s, list):
        pbar = estimator_s
        if verbosity == 1: pbar = tqdm(pbar)
        return np.array([
            apply_get(
                get,
                estimator, gamma_s,
                verbosity,
                **kwargs,
            )
                for estimator in pbar
        ])

    if isinstance(gamma_s, list) or isinstance(gamma_s, np.ndarray):
        pbar = gamma_s
        if verbosity == 2: pbar = tqdm(pbar)
        array = np.array([
            apply_get(
                get,
                estimator_s, gamma,
                verbosity,
                **kwargs,
            )
                for gamma in pbar
        ])

        return array

    return get(estimator_s, gamma_s, **kwargs)


def get_pv_s(
        estimator_s, gamma_s,
        verbosity=0,
        **kwargs,
    ):

    return apply_get(
        get_pv,
        estimator_s, gamma_s,
        verbosity,
        **kwargs,
    )

def get_sdc_s(
        estimator_s, gamma_s,
        verbosity=0,
        **kwargs,
    ):

    return apply_get(
        get_sdc,
        estimator_s, gamma_s,
        verbosity,
        **kwargs,
    )

def get_vaf_s(
        estimator_s, gamma_s,
        verbosity=0,
        **kwargs,
    ):

    return apply_get(
        get_vaf,
        estimator_s, gamma_s,
        verbosity,
        **kwargs,
    )

# ---------------------------------------------------------------- #
