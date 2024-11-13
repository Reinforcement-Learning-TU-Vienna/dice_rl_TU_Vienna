# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC, abstractmethod

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import TimeStep, time_step_spec
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.typing.types import NestedTensorSpec

# ---------------------------------------------------------------- #

class MyTFPolicy(TFPolicy, ABC):
    def __init__(self, observation_spec, action_spec, reward_spec=None):
        super().__init__(
            time_step_spec(observation_spec, reward_spec), action_spec, )

    @abstractmethod
    def _probs(self, time_step):
        pass

    def _logits(self, time_step):
        probs = self._probs(time_step)
        logits = np.log(probs) # type: ignore
        return logits

    def _distribution(
        self, time_step: TimeStep,
        policy_state: NestedTensorSpec) -> PolicyStep:

        logits = self._logits(time_step)
        action = tfp.distributions.Categorical(logits=logits, dtype=tf.int32)

        state = ()
        info = ()

        return PolicyStep(action, state, info)

# ---------------------------------------------------------------- #
