# ---------------------------------------------------------------- #

import warnings

from gymnasium import Wrapper, Env
from typing import Union

# ---------------------------------------------------------------- #

class MyWrapper(Wrapper):
    def reset(self, *, seed=None, options=None):
        reset_done = True
        if options is not None: reset_done = options.get("reset_done", True)
        if reset_done: self.done = False

        return super().reset(seed=seed, options=options)


class AbsorbingWrapper(MyWrapper):
    def __init__(self, env: Env, absorbing_rew: Union[float, None] = None, absorbing_obs=None):
        super().__init__(env)
        self._absorbing_rew = absorbing_rew
        self._absorbing_obs = absorbing_obs

    def step(self, act):
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            obs_next, rew, terminated, truncated, info = self.env.step(act)

            done = terminated or truncated or self.done

            if done:
                if (o := self._absorbing_obs) is not None: obs_next = o
                if (r := self._absorbing_rew) is not None: rew = r

                self.done = True

            terminated = False
            truncated = False
            info["absorbing"] = self.done

            return obs_next, rew, terminated, truncated, info


class LoopingWrapper(MyWrapper):
    def __init__(self, env: Env, looping_rew: Union[float, None] = None):
        super().__init__(env)
        self._looping_rew = looping_rew

    def step(self, act):
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            obs_next, rew, terminated, truncated, info = self.env.step(act)

            done = terminated or truncated

            if done:
                obs_next, i = self.reset(options={"reset_done": False})
                for k, v in i.items():
                    info[k] = v

                if (r := self._looping_rew) is not None: rew = r

                self.done = True

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
