# ---------------------------------------------------------------- #

import torch

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------- #

def get_probs(obs, model, n_act):
    o = obs
    o = np.vstack(o)
    o = torch.tensor(o)

    distribution = model.policy.get_distribution(o)

    logits = np.array([
        distribution.log_prob( torch.tensor(action) ).detach().numpy()
            for action in range(n_act)
    ]).T
    probs = tf.nn.softmax(logits)
    probs = np.array(probs)

    return probs

# ---------------------------------------------------------------- #
