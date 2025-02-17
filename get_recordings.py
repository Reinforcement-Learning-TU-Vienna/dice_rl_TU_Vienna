# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------- #

def flatten_grads(grads):
    f = [ tf.reshape(t, shape=[-1]) for t in grads ]
    f = tf.concat(f, axis=-1)
    return f

# ---------------------------------------------------------------- #

def get_recordings_cos_angle(
        estimator,
        obs_init, obs, act, obs_next, probs_init, probs_next,
        values, loss, gradients,
        pv_s, pv_w, ):

        # estimator,
        # env_steps, values, loss, gradients):

    x = flatten_grads(gradients["v"])
    y = flatten_grads(gradients["v_next"])

    a = np.dot(x, y) # type: ignore
    b = np.linalg.norm(x) * np.linalg.norm(y) # type: ignore

    cos_angle = a / b

    return { "cos_angle": cos_angle, }

# ---------------------------------------------------------------- #
