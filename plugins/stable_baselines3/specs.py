# ---------------------------------------------------------------- #

from gymnasium.spaces import Discrete, Box

# ---------------------------------------------------------------- #

def get_specs_space(space):
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

    return { "shape": shape, "dtype": dtype, "min": min, "max": max, }


def get_specs_env(env):
    return {
        "obs": get_specs_space(env.observation_space),
        "act": get_specs_space(env.action_space),
    }

# ---------------------------------------------------------------- #
