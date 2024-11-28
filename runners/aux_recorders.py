# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------- #

def flatten_grads(grads):
    f = [ tf.reshape(t, shape=[-1]) for t in grads ]
    f = tf.concat(f, axis=-1)
    return f

# ---------------------------------------------------------------- #

def aux_recorder_cos_angle(
        estimator,
        env_steps, values, loss, gradients):

    x = flatten_grads(gradients["primal_values"])
    y = flatten_grads(gradients["primal_values_next"])

    a = np.dot(x, y) # type: ignore
    b = np.linalg.norm(x) * np.linalg.norm(y) # type: ignore

    cos_angle = a / b

    return { "cos_angle": cos_angle, }

# ---------------------------------------------------------------- #
