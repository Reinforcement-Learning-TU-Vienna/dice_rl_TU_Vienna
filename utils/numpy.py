# ---------------------------------------------------------------- #

import numpy as np

import warnings

from tqdm import tqdm

# ---------------------------------------------------------------- #

@np.vectorize
def safe_divide(x, y, zero_div_zero=0):
    if y == 0:
        assert x == 0, f"cannot divide {x=} by {y=}"
        return zero_div_zero
    else:
        return x / y


def group(x, size=1):
    groups = len(x) // size

    s = []

    for i in range(groups):
        k = i * size
        l = min( len(x), k + size )

        s.append( np.sum(x[k:l]) )

    return s


def add_middle_means(x, multiplicity=1):
    assert len(x) >= 2

    y = ( x[1:] + x[:-1] ) / 2

    z = np.zeros(len(x) + len(y))
    z[0::2] = x
    z[1::2] = y

    if multiplicity == 1:
        return z
    else:
        return add_middle_means(z, multiplicity-1)

def add_middle_means_log(x, multiplicity=1):
    l = np.log(x)
    m = add_middle_means(l, multiplicity)
    e = np.exp(m)
    return e


# https://stackoverflow.com/a/54628145/16192280
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def pad(x, c=0, verbosity=0):
    pbar = x
    if verbosity == 1: pbar = tqdm(pbar)
    x = [list(y) for y in pbar]

    len_max = np.max([len(y) for y in x])

    for i, y in enumerate(x):
        l = len(y)
        x[i] += [c] * (len_max - l)

    return np.array(x), len_max


def eye_like(x):
    N, M = x.shape
    return np.eye(N, M)


def is_real(x):
    return np.all(x == np.real(x))

def check_real(x):
    if not is_real(x):
        msg = f"Not all entries are real: {x=}"
        warnings.warn(msg, UserWarning)


def has_unique_sign(x):
    L = np.prod(x <= 0)
    G = np.prod(x >= 0)
    return L or G

def check_unique_sign(x):
    if not has_unique_sign(x):
        msg = f"Not all entries have the same sign: {x=}"
        warnings.warn(msg, UserWarning)


def get_eigenvalue_for_eigenvector_of(matrix, eigenvalue_exact=1):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    index = np.argmin( np.abs(eigenvalues - eigenvalue_exact) )
    eigenvalue  = eigenvalues[index]
    eigenvector = eigenvectors[:, index]

    check_real(eigenvector)
    eigenvector = np.real(eigenvector)

    check_unique_sign(eigenvector)
    eigenvector = np.abs(eigenvector)

    return eigenvalue, eigenvector

# ---------------------------------------------------------------- #

def project_in(mask, vectors_sup, matrices_sup, projected):
    if projected:
        vectors_sub = tuple([
            vector_sub[~mask] for vector_sub in vectors_sup
        ])
        matrices_sub = tuple([
            matrix_sup[~mask, :][:, ~mask] for matrix_sup in matrices_sup
        ])

    else:
        for i, vector_sup in enumerate(vectors_sup):
            assert np.sum(vector_sup[mask] != 0) == 0, i

        for i, matrix_sup in enumerate(matrices_sup):
            assert np.sum(matrix_sup[mask, :] != 0) == 0, i
            assert np.sum(matrix_sup[:, mask] != 0) == 0, i

        vectors_sub = vectors_sup
        matrices_sub = matrices_sup

    return vectors_sub, matrices_sub


def project_out(mask, sub, projected, masking_value):
    if projected:
        sup = np.zeros_like(mask, dtype=float)
        sup[~mask] = sub
        sup[ mask] = masking_value

    else:
        sup = sub

    return sup

# ---------------------------------------------------------------- #
