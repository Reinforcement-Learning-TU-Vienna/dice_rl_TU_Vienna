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

def get_pv_OnPE(estimator, gamma, *args):
    pv = estimator(gamma)
    return pv

def get_pv_OffPE(estimator, gamma, *args):
    projected, weighted, modified, lamda = args

    pv, *_ = estimator.solve(
        gamma, projected,
        weighted=weighted,
        modified=modified, lamda=lamda,
    )
    return pv

def get_sdc(estimator, gamma, *args):
    projected, modified, lamda = args

    sdc, _ = estimator.solve_sdc(
        gamma, projected,
        modified=modified, lamda=lamda,
    )
    return sdc

def get_vaf(estimator, gamma, *args):
    projected, = args

    vaf, _ = estimator.solve_vaf(
        gamma, projected,
    )
    return vaf


def apply_get(
        get, file_name,
        estimator_s, gamma_s,
        verbosity=0,
        *args):

    if isinstance(estimator_s, list):
        pbar = estimator_s
        if verbosity == 1: pbar = tqdm(pbar)
        return np.array([
            apply_get(
                get, file_name,
                estimator, gamma_s,
                verbosity,
                *args
            )
                for estimator in pbar
        ])

    if isinstance(gamma_s, list) or isinstance(gamma_s, np.ndarray):
        pbar = gamma_s
        if verbosity == 2: pbar = tqdm(pbar)
        array = np.array([
            apply_get(
                get, file_name,
                estimator_s, gamma,
                verbosity,
                *args
            )
                for gamma in pbar
        ])

        return array

    return get(estimator_s, gamma_s, *args)


def get_pv_s_OnPE(
        estimator_s, gamma_s,
        verbosity=0, ):

    return apply_get(
        get_pv_OnPE, "pv",
        estimator_s, gamma_s,
        verbosity, )

def get_pv_s_OffPE(
        estimator_s, gamma_s,
        projected, weighted, modified, lamda,
        verbosity=0, ):

    return apply_get(
        get_pv_OffPE, "pv",
        estimator_s, gamma_s,
        verbosity,
        projected, weighted, modified, lamda, )

def get_sdc_s(
        estimator_s, gamma_s,
        projected, modified, lamda,
        verbosity=0, ):

    return apply_get(
        get_sdc, "sdc",
        estimator_s, gamma_s,
        verbosity,
        projected, modified, lamda, )

def get_vaf_s(
        estimator_s, gamma_s,
        projected,
        verbosity=0, ):

    return apply_get(
        get_vaf, "vaf",
        estimator_s, gamma_s,
        verbosity,
        projected, )

# ---------------------------------------------------------------- #
