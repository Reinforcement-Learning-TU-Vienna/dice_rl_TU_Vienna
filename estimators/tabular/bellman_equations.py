# ---------------------------------------------------------------- #

import numpy as np

from utils.numpy import (
    safe_divide, eye_like, get_eigenvalue_for_eigenvector_of,
    project_in, project_out, )

# ---------------------------------------------------------------- #

def solve_forwards_bellman_equations(
        dD, r, P, gamma, projected=False):

    mask = dD == 0

    (r_,), (P_,) = project_in(mask, (r,), (P,), projected)

    f, info = solve_standard_forwards_bellman_equations(
        r=r_, P=P_, gamma=gamma)

    return f, info


def solve_standard_forwards_bellman_equations(
        r, P, gamma):

    if gamma < 1:
        I = eye_like(P)

        a = I - gamma * P
        b = r

        f = np.linalg.solve(a, b)
        info = {}

    else:
        f = None
        info = { "pv": None, }

        raise NotImplementedError

    return f, info


def solve_forwards_bellman_equations_approximate(
        dD_bar, r_bar, P_bar, gamma, projected=False):

    mask = dD_bar == 0

    (dD_, r_,), (P_,) = project_in(mask, (dD_bar, r_bar,), (P_bar,), projected)

    f_, info = solve_standard_forwards_bellman_equations_approximate(
        dD_bar=dD_, r_bar=r_, P_bar=P_, gamma=gamma)

    f = project_out(mask, f_, projected, masking_value=0)

    return f, info


def solve_standard_forwards_bellman_equations_approximate(
        dD_bar, r_bar, P_bar, gamma):

    r_hat = safe_divide(r_bar, dD_bar)
    P_hat = safe_divide(P_bar.T, dD_bar).T

    f_hat, info = solve_standard_forwards_bellman_equations(
        r=r_hat, P=P_hat, gamma=gamma, )

    if "pv" in info.keys(): info = { "pv_hat": info["pv"], }

    return f_hat, info

# ---------------------------------------------------------------- #

def solve_backwards_bellman_equations(
        d0, dD, P, gamma, modified=False, projected=False):

    mask = dD == 0

    (d0_, dD_), (P_,) = project_in(mask, (d0, dD), (P,), projected)

    if modified:
        sdc_, info = solve_modified_backwards_bellman_equations(
            d0=d0_, dD=dD_, P=P_, gamma=gamma)

    else:
        sd_, info = solve_standard_backwards_bellman_equations(
            d0=d0_, P=P_, gamma=gamma)
        sdc_ = safe_divide(sd_, dD_, zero_div_zero=-1)

    sdc = project_out(mask, sdc_, projected, masking_value=-1)

    assert np.prod( (sdc == -1) == mask )

    return sdc, info


def solve_standard_backwards_bellman_equations(
        d0, P, gamma):

    if gamma < 1:
        I = eye_like(P)

        a = I - gamma * P.T
        b = (1 - gamma) * d0

        sd = np.linalg.solve(a, b)
        info = {}

    else:
        ev, sd = get_eigenvalue_for_eigenvector_of(P.T)
        sd /= np.linalg.norm(sd, ord=1)
        info = { "ev": ev, }

    return sd, info

def solve_modified_backwards_bellman_equations(
        d0, dD, P, gamma):

    mask = dD == 0

    D = np.diag(dD)

    if gamma < 1:
        a = D - gamma * P.T @ D
        b = (1 - gamma) * d0

        a += np.diag(mask)
        b -= mask

        sdc = np.linalg.solve(a, b)
        info = {}

    else:
        assert np.prod(~mask)
        D_inv = np.diag(1 / dD)

        ev, sdc = get_eigenvalue_for_eigenvector_of(
            D_inv @ P.T @ D)
        sdc /= np.dot(dD, sdc)
        info = { "ev": ev, }

    return sdc, info


def solve_backwards_bellman_equations_approximate(
        d0_bar, dD_bar, P_bar, n, gamma, modified=False, projected=False):

    mask = dD_bar == 0

    (d0_, dD_), (P_,) = project_in(mask, (d0_bar, dD_bar), (P_bar,), projected)

    if modified:
        sdc_, info = solve_modified_backwards_bellman_equations_approximate(
            d0_bar=d0_, dD_bar=dD_, P_bar=P_, n=n, gamma=gamma)

    else:
        sd_, info = solve_standard_backwards_bellman_equations_approximate(
            d0_bar=d0_, dD_bar=dD_, P_bar=P_, n=n, gamma=gamma)
        sdc_ = safe_divide(sd_, dD_ / n, zero_div_zero=-1)

    sdc_hat = project_out(mask, sdc_, projected, masking_value=-1)

    assert np.prod( (sdc_hat == -1) == mask )

    return sdc_hat, info


def solve_standard_backwards_bellman_equations_approximate(
        d0_bar, dD_bar, P_bar, n, gamma):

    d0_hat = d0_bar / n
    P_hat = safe_divide(P_bar.T, dD_bar).T

    sd_hat, info = solve_standard_backwards_bellman_equations(
        d0=d0_hat, P=P_hat, gamma=gamma, )

    if "ev" in info.keys(): info = { "ev_hat": info["ev"], }

    return sd_hat, info

def solve_modified_backwards_bellman_equations_approximate(
        d0_bar, dD_bar, P_bar, n, gamma):

    mask = dD_bar == 0

    D_bar = np.diag(dD_bar)

    if gamma < 1:
        a = D_bar - gamma * P_bar.T
        b = (1 - gamma) * d0_bar

        a += np.diag(mask)
        b -= mask

        sdc_hat = np.linalg.solve(a, b)
        info = {}

    else:
        ev_hat, sdc_hat = get_eigenvalue_for_eigenvector_of(
            safe_divide(P_bar, dD_bar).T )
        sdc_hat /= np.dot(dD_bar, sdc_hat)
        sdc_hat *= n
        info = { "ev_hat": ev_hat, }

    return sdc_hat, info

# ---------------------------------------------------------------- #

def test_avf(gamma, Q, P, r):
    assert 0 < gamma < 1

    title = f"Checking forwards Bellman equations ({gamma=}):"
    bar = "-" * len(title)
    print(title); print(bar)

    print("Q^pi = r + gamma * P^pi Q^pi")
    lhs = Q
    rhs = r + gamma * P @ Q
    print("->", f"MSE = {np.mean( (rhs - lhs) ** 2 )}")

    print()


def test_sd(gamma, d, d0, P):

    title = f"Checking backwards Bellman equations ({gamma=}):"
    bar = "-" * len(title)
    print(title); print(bar)

    lhs = d
    rhs = (1 - gamma) * d0 + gamma * P.T @ d
    if gamma < 1:
        print("d^pi = (1 - gamma) * d_0^pi + gamma * P^pi_* d^pi")
    else:
        print("d^pi = P^pi_* d^pi")
    print("->", f"MSE = {np.mean( (rhs - lhs) ** 2 )}")

    print("d^pi >= 0")
    print("->", np.all(d >= 0) )

    print("sum(d^pi) = 1")
    print("->", f"sum = {np.sum(d)}")

    print()


def test_sdc(gamma, w, d0, dD, P):

    title = f"Checking modified backwards Bellman equations ({gamma=}):"
    bar = "-" * len(title)
    print(title); print(bar)

    DD = np.diag(dD)

    lhs = DD @ w
    rhs = (1 - gamma) * d0 + gamma * P.T @ DD @ w
    if gamma < 1:
        print("D^D w = (1 - gamma) * d_0^pi + gamma * P^pi_* D^D w")
    else:
        print("d^pi = P^pi_* d^pi")
    print("->", f"MSE = {np.mean( (rhs - lhs) ** 2 )}")

    print("w >= 0")
    print("->", np.all(w >= 0) )

    print("sum(w * d^D) = 1")
    print("->", f"sum = {np.sum(w * dD)}")

    print()

# ---------------------------------------------------------------- #
