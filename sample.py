# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from dice_rl.data.dataset import convert_to_tfagents_timestep

# ---------------------------------------------------------------- #

def get_probs(env_step, policy=None):
    if policy is not None:
        env_step_tf = convert_to_tfagents_timestep(env_step)
        env_step_target_probs = policy \
            .distribution(env_step_tf) \
            .action \
            .probs_parameter() # type: ignore
    else:
        assert "probability" in env_step.other_info.keys()
        env_step_target_probs = env_step.other_info["probability"]

    env_step_target_probs = tf.cast(env_step_target_probs, dtype=tf.float32)
    return env_step_target_probs


def get_probs_log(env_step, policy=None):
    return np.array([
        np.log(prob) if prob > 0 else -1e-8
            for prob in get_probs(env_step, policy) # type: ignore
    ])

# ---------------------------------------------------------------- #
