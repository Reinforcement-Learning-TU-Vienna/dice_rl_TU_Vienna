# ---------------------------------------------------------------- #

import warnings

from gymnasium import Wrapper, Env
from typing import Union

# ---------------------------------------------------------------- #

class AbsorbingWrapper(Wrapper):
    def __init__(self, env: Env, absorbing_rew: Union[float, None] = None, absorbing_obs=None):
        super().__init__(env)
        self._absorbing_rew = absorbing_rew
        self._absorbing_obs = absorbing_obs

    def step(self, act):
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            obs_next, rew, terminated, truncated, info = self.env.step(act)

            done = terminated or truncated

            if done:
                if (o := self._absorbing_obs) is not None: obs_next = o
                if (r := self._absorbing_rew) is not None: rew = r

            terminated = False
            truncated = False
            info["absorbing"] = done

            return obs_next, rew, terminated, truncated, info


class LoopingWrapper(Wrapper):
    def __init__(self, env: Env, looping_rew: Union[float, None] = None):
        super().__init__(env)
        self._looping_rew = looping_rew

    def step(self, act):
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            obs_next, rew, terminated, truncated, info = self.env.step(act)

            done = terminated or truncated

            if done:
                obs_next, i = self.reset()
                for k, v in i.items():
                    info[k] = v

                if (r := self._looping_rew) is not None: rew = r

            terminated = False
            truncated = False
            info["looping"] = done

            return obs_next, rew, terminated, truncated, info

# ---------------------------------------------------------------- #

from gym import Wrapper as WrapperGym

class GymToGymnasiumWrapper(WrapperGym):
    def reset(self, **kwargs):
        obs_init = super().reset(**kwargs)
        info = {}

        return obs_init, info

    def step(self, action):
        obs_next, rew, done, info = super().step(action)

        terminated = done
        truncated = False

        return obs_next, rew, terminated, truncated, info

# ---------------------------------------------------------------- #
