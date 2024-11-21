# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.specs import get_observation_action_spec

from gymnasium.spaces import Discrete, Box

# ---------------------------------------------------------------- #

def get_space_specs(space):
    shape = space.shape
    dtype = space.dtype
 
    min  = None
    max = None

    if isinstance(space, Discrete):
        min = 0
        max = space.n - 1

    if isinstance(space, Box):
        min = space.low
        max = space.high

    assert min is not None and max is not None

    return shape, dtype, min, max


def get_observation_action_spec_from_env(env):
    obs_shape, obs_dtype, obs_min, obs_max = get_space_specs(env.observation_space)
    act_shape, act_dtype, act_min, act_max = get_space_specs(env.action_space)

    return get_observation_action_spec(
        obs_shape, obs_dtype, obs_min, obs_max,
        act_shape, act_dtype, act_min, act_max,
    )

# ---------------------------------------------------------------- #
