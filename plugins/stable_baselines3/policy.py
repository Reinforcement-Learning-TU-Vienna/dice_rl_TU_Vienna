# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

import torch

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from dice_rl_TU_Vienna.policy import MyTFPolicy
from dice_rl_TU_Vienna.plugins.stable_baselines3.specs import get_observation_action_spec_from_env

# ---------------------------------------------------------------- #

class TFPolicyPPO(MyTFPolicy):
    def __init__(self, model, observation_spec, action_spec, reward_spec=None):
        self.model = model
        super().__init__(observation_spec, action_spec, reward_spec)

    def _probs(self, time_step):
        logits = self._logits(time_step)
        return tf.nn.softmax(logits)

    def _logits_distribution(self, observation):
        return self.model.policy.get_distribution(observation)

    def _logits(self, time_step):
        x = time_step.observation

        if tf.rank(x) == 0: x = tf.reshape(x, shape=[1, 1])

        x = np.array(x)
        x = torch.tensor(x)
        observation = x

        distribution = self._logits_distribution(observation)
 
        logits = np.array([
            distribution.log_prob( torch.tensor(action) ).detach().numpy()
                for action in range(self.action_spec.maximum + 1) # type: ignore
        ]).T
        logits = np.squeeze(logits)

        return logits


class TFPolicyMaskablePPO(TFPolicyPPO):
    def __init__(self, model, action_masks, observation_spec, action_spec, reward_spec=None):
        self.action_masks = action_masks
        super().__init__(model, observation_spec, action_spec, reward_spec)

    def _logits_distribution(self, observation):
        action_mask = self.action_masks[observation]
        return self.model.policy.get_distribution(observation, action_mask)

# ---------------------------------------------------------------- #

def get_TFPolicyPPO(env, model):
    return TFPolicyPPO(
        model, *get_observation_action_spec_from_env(env), )

def get_TFPolicyMaskablePPO(env, model, action_masks):
    return TFPolicyMaskablePPO(
        model, action_masks, *get_observation_action_spec_from_env(env), )

# ---------------------------------------------------------------- #

def load_or_create_model(
        model_type,
        model_dir, env=None, total_timesteps=None):

    try:
        print("Try loading model", model_dir)
        model = model_type.load(model_dir)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        assert False

    except:
        assert env is not None
        assert total_timesteps is not None

        print(); print(f"No model found in {model_dir}")

        model = model_type("MlpPolicy", env, verbose=1)

        model.learn(total_timesteps=total_timesteps, tb_log_name="ppo_run")
        model.save(model_dir)

    return model


def load_or_create_model_PPO(model_dir, env=None, total_timesteps=None):
    return load_or_create_model(PPO, model_dir, env, total_timesteps)


def load_or_create_model_MaskablePPO(model_dir, env=None, total_timesteps=None):
    return load_or_create_model(MaskablePPO, model_dir, env, total_timesteps)

# ---------------------------------------------------------------- #
