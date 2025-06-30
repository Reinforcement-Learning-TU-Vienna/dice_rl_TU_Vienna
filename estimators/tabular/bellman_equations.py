# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.utils.numpy import (
    safe_divide, eye_like, get_eigenvalue_for_eigenvector_of,
    project_in, project_out, )

# ---------------------------------------------------------------- #

def solve_forward_bellman_equations(
        dD, r, P, gamma, projected=False):

    mask = dD == 0

    (r_,), (P_,) = project_in(mask, (r,), (P,), projected)

    f, info = solve_standard_forward_bellman_equations(
        r=r_, P=P_, gamma=gamma)

    return f, info


def solve_standard_forward_bellman_equations(
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


def solve_forward_bellman_equations_approximate(
        dD_bar, r_bar, P_bar, gamma, projected=False):

    mask = dD_bar == 0

    (dD_, r_,), (P_,) = project_in(mask, (dD_bar, r_bar,), (P_bar,), projected)

    f_, info = solve_standard_forward_bellman_equations_approximate(
        dD_bar=dD_, r_bar=r_, P_bar=P_, gamma=gamma)

    f = project_out(mask, f_, projected, masking_value=0)

    return f, info


def solve_standard_forward_bellman_equations_approximate(
        dD_bar, r_bar, P_bar, gamma):

    r_hat = safe_divide(r_bar, dD_bar)
    P_hat = safe_divide(P_bar.T, dD_bar).T

    f_hat, info = solve_standard_forward_bellman_equations(
        r=r_hat, P=P_hat, gamma=gamma, )

    if "pv" in info.keys(): info = { "pv_approx": info["pv"], }

    return f_hat, info

# ---------------------------------------------------------------- #

def solve_backward_bellman_equations(
        d0, dD, P, gamma, modified=False, projected=False):

    mask = dD == 0

    (d0_, dD_), (P_,) = project_in(mask, (d0, dD), (P,), projected)

    if modified:
        w_, info = solve_modified_backward_bellman_equations(
            d0=d0_, dD=dD_, P=P_, gamma=gamma)

    else:
        d_, info = solve_standard_backward_bellman_equations(
            d0=d0_, P=P_, gamma=gamma)
        w_ = safe_divide(d_, dD_, zero_div_zero=-1)

    w = project_out(mask, w_, projected, masking_value=-1)

    assert np.prod( (w == -1) == mask )

    return w, info


def solve_standard_backward_bellman_equations(
        d0, P, gamma):

    if gamma < 1:
        I = eye_like(P)

        a = I - gamma * P.T
        b = (1 - gamma) * d0

        d = np.linalg.solve(a, b)
        info = {}

    else:
        l, d = get_eigenvalue_for_eigenvector_of(P.T)
        d /= np.linalg.norm(d, ord=1)
        info = { "ev": l, }

    return d, info

def solve_modified_backward_bellman_equations(
        d0, dD, P, gamma):

    mask = dD == 0

    D = np.diag(dD)

    if gamma < 1:
        a = D - gamma * P.T @ D
        b = (1 - gamma) * d0

        a += np.diag(mask)
        b -= mask

        w = np.linalg.solve(a, b)
        info = {}

    else:
        assert np.prod(~mask)
        D_inv = np.diag(1 / dD)

        l, w = get_eigenvalue_for_eigenvector_of(
            D_inv @ P.T @ D)
        w /= np.dot(dD, w)
        info = { "ev": l, }

    return w, info


def solve_backward_bellman_equations_approximate(
        d0_bar, dD_bar, P_bar, n, gamma, modified=False, projected=False):

    mask = dD_bar == 0

    (d0_, dD_), (P_,) = project_in(mask, (d0_bar, dD_bar), (P_bar,), projected)

    if modified:
        w_, info = solve_modified_backward_bellman_equations_approximate(
            d0_bar=d0_, dD_bar=dD_, P_bar=P_, n=n, gamma=gamma)

    else:
        d_, info = solve_standard_backward_bellman_equations_approximate(
            d0_bar=d0_, dD_bar=dD_, P_bar=P_, n=n, gamma=gamma)
        w_ = safe_divide(d_, dD_ / n, zero_div_zero=-1)

    w_hat = project_out(mask, w_, projected, masking_value=-1)

    assert np.prod( (w_hat == -1) == mask )

    return w_hat, info


def solve_standard_backward_bellman_equations_approximate(
        d0_bar, dD_bar, P_bar, n, gamma):

    d0_hat = d0_bar / n
    P_hat = safe_divide(P_bar.T, dD_bar).T

    d, info = solve_standard_backward_bellman_equations(
        d0=d0_hat, P=P_hat, gamma=gamma, )

    if "ev" in info.keys(): info = { "ev_approx": info["ev"], }

    return d, info

def solve_modified_backward_bellman_equations_approximate(
        d0_bar, dD_bar, P_bar, n, gamma):

    mask = dD_bar == 0

    D_bar = np.diag(dD_bar)

    if gamma < 1:
        a = D_bar - gamma * P_bar.T
        b = (1 - gamma) * d0_bar

        a += np.diag(mask)
        b -= mask

        w_hat = np.linalg.solve(a, b)
        info = {}

    else:
        l_hat, w_hat = get_eigenvalue_for_eigenvector_of(
            safe_divide(P_bar, dD_bar).T )
        w_hat /= np.dot(dD_bar, w_hat)
        w_hat *= n
        info = { "ev_approx": l_hat, }

    return w_hat, info

# ---------------------------------------------------------------- #

def test_avf(gamma, Q, P, r):
    assert 0 < gamma < 1

    title = f"Checking forward Bellman equations ({gamma=}):"
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
