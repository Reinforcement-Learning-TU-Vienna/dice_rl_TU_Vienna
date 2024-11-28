# ---------------------------------------------------------------- #

import numpy as np

from tf_agents.specs import tensor_spec, BoundedArraySpec

# ---------------------------------------------------------------- #

def get_observation_action_spec(
        obs_shape, obs_dtype, obs_min, obs_max,
        act_shape, act_dtype, act_min, act_max,
    ):

    observation_spec = tensor_spec.from_spec(
        BoundedArraySpec(
            shape=obs_shape,
            dtype=obs_dtype,
            minimum=np.array(obs_min, dtype=obs_dtype),
            maximum=np.array(obs_max, dtype=obs_dtype),
            name="observation"))

    action_spec = tensor_spec.from_spec(
        BoundedArraySpec(
            shape=act_shape,
            dtype=act_dtype,
            minimum=np.array(act_min, dtype=act_dtype),
            maximum=np.array(act_max, dtype=act_dtype),
            name="action"))

    return observation_spec, action_spec

def get_step_num_spec(step_num_max):
    return tensor_spec.from_spec(
        BoundedArraySpec(
            shape=(),
            dtype=np.int64,
            minimum=0,
            maximum=step_num_max,
            name="step_num"))

# ---------------------------------------------------------------- #

def get_observation_action_spec_tabular(bounds):
    obs_min, obs_max, act_min, act_max = bounds

    return get_observation_action_spec(
        obs_shape=(), act_shape=(),
        obs_dtype=np.int64, act_dtype=np.int64,
        obs_min=obs_min, obs_max=obs_max, act_min=act_min, act_max=act_max,
    )

def get_observation_action_spec_continuous(bounds, shapes):
    obs_min, obs_max, act_min, act_max = bounds
    obs_shape, act_shape = shapes

    return get_observation_action_spec(
        obs_shape=obs_shape, act_shape=act_shape,
        obs_dtype=np.float32, act_dtype=np.int64,
        obs_min=obs_min, obs_max=obs_max, act_min=act_min, act_max=act_max,
    )

# ---------------------------------------------------------------- #
