# ---------------------------------------------------------------- #

import os

import torch

import numpy as np
import tensorflow as tf

from datetime import datetime

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from dice_rl_TU_Vienna.utils.seeds import set_all_seeds
from dice_rl_TU_Vienna.utils.json import (
    json_get_id, json_remove_by_dict, json_append, )

# ---------------------------------------------------------------- #

def create_model_from_env(model_type, env, hyperparameters):
    set_all_seeds(hyperparameters["seed"])

    model = model_type("MlpPolicy", env, verbose=1)
    model.learn(
        total_timesteps=hyperparameters["total_timesteps"],
        tb_log_name="ppo_run",
    )

    return model


def save_model(dir_data, model, hyperparameters, verbosity=0):
    file_path_json = os.path.join(dir_data, "policy.json")

    id_policy = datetime.now().isoformat()
    dir_policy = os.path.join(dir_data, id_policy)
    file_path_zip = os.path.join(dir_policy, "policy.zip")

    hyperparameters_labeled = {
        "id": id_policy,
        "data": hyperparameters,
    }

    json_append(file_path=file_path_json, dictionary=hyperparameters_labeled)

    if not tf.io.gfile.isdir(dir_policy):
        tf.io.gfile.makedirs(dir_policy)

    model.save(file_path_zip)
    if verbosity > 0: print(f"saved {file_path_zip}")

    return id_policy


def get_model(model_type, dir_data, env, hyperparameters, verbosity=0):
    file_path_json = os.path.join(dir_data, "policy.json")
    loaded = False

    if verbosity > 0: print(f"trying to find id_policy in {file_path_json}")
    id_policy = json_get_id(file_path=file_path_json, dictionary=hyperparameters)

    if id_policy is not None:
        dir_policy = os.path.join(dir_data, id_policy)
        file_path_zip = os.path.join(dir_policy, "policy.zip")

        if verbosity > 0: print(f"trying to load policy from {file_path_zip}")
        if os.path.exists(file_path_zip):
            model = model_type.load(file_path_zip)
            loaded = True

    if not loaded:
        if verbosity > 0: print("failed to load")
        if verbosity > 0: print("creating policy from environment")

        model = create_model_from_env(model_type, env, hyperparameters)

        if verbosity > 0: print(f"removing policy hyperparameters from {file_path_json}")
        json_remove_by_dict(file_path=file_path_json, dictionary=hyperparameters)

        if verbosity > 0: print("saving dataset")
        id_policy = save_model(dir_data, model, hyperparameters, verbosity)

    return model, id_policy

def get_model_PPO(dir_data, env, hyperparameters, verbosity=0):
    model_type = PPO
    hyperparameters["model_type"] = "PPO"
    return get_model(model_type, dir_data, env, hyperparameters, verbosity)

def get_model_MaskablePPO(dir_data, env, hyperparameters, verbosity=0):
    model_type = MaskablePPO
    hyperparameters["model_type"] = "MaskablePPO"
    return get_model(model_type, dir_data, env, hyperparameters, verbosity)


def get_probs(obs, get_distribution, n_act):
    o = obs
    o = np.array(o)
    if tf.rank(o) == 0: o = tf.reshape(o, shape=[1, 1])
    o = torch.tensor(o)

    distribution = get_distribution(o)

    logits = np.array([
        distribution.log_prob( torch.tensor(action) ).detach().numpy()
            for action in range(n_act)
    ]).T
    probs = tf.nn.softmax(logits)
    probs = np.array(probs)

    return probs

def get_probs_PPO(obs, model, n_act):
    get_distribution = lambda o: model.policy.get_distribution(o)

    return get_probs(obs, get_distribution, n_act)

def get_probs_MaskablePPO(obs, model, action_masks):
    n_obs, n_act = action_masks.shape
    get_distribution = lambda o: model.policy.get_distribution(o, action_masks[o])

    return get_probs(obs, get_distribution, n_act)

# ---------------------------------------------------------------- #
