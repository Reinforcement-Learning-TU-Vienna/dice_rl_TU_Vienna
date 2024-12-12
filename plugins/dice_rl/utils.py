import numpy as np
import tensorflow.compat.v2 as tf # type: ignore

def get_mean_neural_coin_dice(running_losses):

    mean = [ # look at output of train_step in neural_coin_dice.py
        np.zeros(2), # estimate
        np.zeros(4), # weighted_nu_loss
        np.zeros(4), # weighted_zeta_loss
        np.zeros(2), # weight_loss
        np.zeros(2), # divergence
    ]

    n = len(running_losses)
    m = len(mean)

    for k in range(n):
        loss = running_losses[k]
        for i in range(m):
            mean[i] += loss[i]

    for i in range(m):
        mean[i] /= n

    return mean
