# ---------------------------------------------------------------- #

import numpy as np

# ---------------------------------------------------------------- #

def get_w_prod(dataset, weights):
    w_prod = []

    for id in dataset["id"].unique():
        f = dataset["id"] == id
        w = np.prod(weights[f]) if weights is not None else 1
        w_prod.append(w)

    w_prod = np.array(w_prod)

    return w_prod

def get_H(dataset):
    _, H = np.unique(dataset["id"], return_counts=True)

    return H

def get_get_policy_value(dataset, weights=None):
    r = dataset["rew"]
    t = dataset["t"]
    n = dataset["id"].unique().size

    w_prod = get_w_prod(dataset, weights)
    H = get_H(dataset)

    def get_policy_value(gamma, scale=True, weighted=False):
        if weighted: assert weights is not None

        if gamma < 1:
            f = (1 - gamma) if scale else 1
            R = f * np.sum( r * gamma ** t * np.repeat(w_prod, H) )

        else:
            f = 1 / np.repeat(H, H) if scale else 1
            R = np.sum( f * r * np.repeat(w_prod, H) )

        x = R
        y = n if not weighted else np.sum(w_prod)
        return x / y

    return get_policy_value

def get_success_rate(dataset):
    r = dataset["rew"]
    n = dataset["id"].unique().size

    return np.sum(r) /  n

# ---------------------------------------------------------------- #
