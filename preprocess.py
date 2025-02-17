# ---------------------------------------------------------------- #

import numpy as np

# ---------------------------------------------------------------- #

def one_hot_encode_observation(estimator, obs):

    assert len(estimator.obs_shape) == 1
    n_obs, *_ = estimator.obs_shape

    I = np.identity(n_obs)
    obs_OHC = I[obs]

    return obs_OHC

# ---------------------------------------------------------------- #
