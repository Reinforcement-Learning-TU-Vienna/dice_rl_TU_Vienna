# ---------------------------------------------------------------- #

import os
import warnings

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime

from tensorflow.keras.utils import set_random_seed # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

from dice_rl.networks.value_network import ValueNetwork

from dice_rl_TU_Vienna.specs import get_observation_action_spec_continuous
from dice_rl_TU_Vienna.utils.json import json_append
from dice_rl_TU_Vienna.utils.tensorflow import learning_rate_hyperparameter
from dice_rl_TU_Vienna.utils.seeds import set_all_seeds

# ---------------------------------------------------------------- #

# warnings.simplefilter("ignore", SyntaxWarning)

# ---------------------------------------------------------------- #

class NeuralDice(ABC):
    def __init__(
            self,
            gamma, seed, batch_size,
            learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs=None, preprocess_act=None, preprocess_rew=None,
            dir=None, get_recordings=None, other_hyperparameters=None, save_interval=100):

        if preprocess_obs is None: preprocess_obs = lambda estimator, obs: self.preprocess_probs(obs)
        if preprocess_act is None: preprocess_act = lambda estimator, act: act
        if preprocess_rew is None: preprocess_rew = lambda estimator, rew: rew

        if get_recordings is None:
            get_recordings = lambda *args: {}

        self.gamma = gamma
        self.seed = seed
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.hidden_dimensions = hidden_dimensions
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.n_act = n_act
        self.obs_shape = obs_shape

        self.dataset = dataset
        self._preprocess_obs = preprocess_obs
        self._preprocess_act = preprocess_act
        self._preprocess_rew = preprocess_rew

        self.dir = dir
        self.get_recordings = get_recordings
        self.save_interval = save_interval

        self.hyperparameters = {
            "name": self.__name__,
            "gamma": gamma,
            "seed": seed,
            "batch_size": batch_size,
            "learning_rate": learning_rate_hyperparameter(learning_rate),
            "hidden_dimensions": hidden_dimensions,
        }
        if other_hyperparameters is not None:
            self.hyperparameters["other"] = other_hyperparameters

        set_all_seeds(self.seed)

        self.set_up_recording()
        self.set_up_networks()

    @property
    @abstractmethod
    def __name__(self): pass

    @property
    def save_dir(self):
        if self.dir is None: return None
        return os.path.join(self.dir, self.id)

    def set_up_recording(self):
        if self.dir is None:
            tf.summary.create_noop_writer()
            return

        self.id = datetime.now().isoformat()

        self.save_hyperparameters()

        summary_writer = tf.summary.create_file_writer(logdir=self.save_dir)
        summary_writer.set_as_default()

    def save_hyperparameters(self):
        assert self.dir is not None

        file_dir = os.path.join(self.dir, "evaluation.json")

        hyperparameters_labeled = {
            "id": self.id,
            "data": self.hyperparameters,
        }

        json_append(file_dir, hyperparameters_labeled)

    def preprocess_obs(self, obs): return tf.convert_to_tensor(self._preprocess_obs(self, obs), dtype=tf.float32)
    def preprocess_act(self, act): return tf.convert_to_tensor(self._preprocess_act(self, act), dtype=tf.int64)
    def preprocess_rew(self, rew): return tf.convert_to_tensor(self._preprocess_rew(self, rew), dtype=tf.float32)

    def preprocess_probs(self, probs):
        x = probs
        x = np.array(probs)
        x = np.vstack(x) # type: ignore
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x

    @property
    def output_activation_fn_dual(self): return tf.identity

    def set_up_networks(self):

        input_tensor_spec = get_observation_action_spec_continuous(
            self.obs_min, self.obs_max, self.n_act, self.obs_shape, )

        self.v = ValueNetwork(
            input_tensor_spec,
            fc_layer_params=self.hidden_dimensions,
        )

        self.w = ValueNetwork(
            input_tensor_spec,
            fc_layer_params=self.hidden_dimensions,
            output_activation_fn=self.output_activation_fn_dual,
        )

        self.u = None

        self.v.create_variables()
        self.w.create_variables()

        self.v_optimizer = SGD(self.learning_rate)
        self.w_optimizer = SGD(self.learning_rate)

    def get_value(self, network, obs, act):
        inputs = obs, act
        value, _ = network(inputs)
        return value

    def get_average_value(self, network, obs, probs):
        batch_size = obs.shape[0]
        obs_shape = obs.shape[1:]
        n_actions = probs.shape[1]

        input = tf.expand_dims(obs, axis=1)
        multiples = [1, n_actions] + [1] * len(obs_shape)
        tensor = tf.tile(input, multiples)

        act_space = tf.range(n_actions, dtype=tf.int64)

        obs_tiled = tf.reshape(tensor, [batch_size * n_actions] + obs_shape)
        act_tiled = tf.tile(act_space, [batch_size])
        probs_flat = tf.reshape(probs, [-1])

        values = self.get_value(network, obs_tiled, act_tiled)

        x = values * probs_flat
        x = tf.reshape(x, [batch_size, n_actions])
        x = tf.reduce_sum(x, axis=1)
        average_values = x

        return average_values

    def get_values(
            self,
            obs_init, obs, act, obs_next,
            probs_init, probs_next):

        v_init = self.get_average_value(self.v, obs_init, probs_init)
        v      = self.get_value        (self.v, obs, act)
        v_next = self.get_average_value(self.v, obs_next, probs_next)
        w      = self.get_value        (self.w, obs, act)

        return v_init, v, v_next, w

    @abstractmethod
    def get_loss(self, v_init, v, v_next, w):
        raise NotImplementedError

    @tf.function(jit_compile=True)
    def get_gradients(
            self,
            obs_init, obs, act, obs_next,
            probs_init, probs_next,
            batch_size):

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch([variable.value for variable in self.v.variables])
            tape.watch([variable.value for variable in self.w.variables])
            if self.u is not None: tape.watch([self.u])

            values = self.get_values(
                obs_init, obs, act, obs_next,
                probs_init, probs_next, )

            loss = self.get_loss(*values)

            v_loss = -loss
            w_loss =  loss
            u_loss = -loss

        gradients = {}

        # ---- #
        # values

        v_init, v, v_next, w = values

        grads_v_init = tape.gradient(v_init, self.v.variables)
        grads_v      = tape.gradient(v,      self.v.variables)
        grads_v_next = tape.gradient(v_next, self.v.variables)
        grads_w      = tape.gradient(w,      self.w.variables)

        grads_v_init = [ grad / batch_size for grad in grads_v_init ] # type: ignore
        grads_v      = [ grad / batch_size for grad in grads_v ]      # type: ignore
        grads_v_next = [ grad / batch_size for grad in grads_v_next ] # type: ignore
        grads_w      = [ grad / batch_size for grad in grads_w ]      # type: ignore

        gradients["v_init"] = grads_v_init
        gradients["v"]      = grads_v
        gradients["v_next"] = grads_v_next
        gradients["w"]      = grads_w

        # ---- #
        # loss

        grads_v_loss = tape.gradient(v_loss, self.v.variables)
        grads_w_loss = tape.gradient(w_loss, self.w.variables)

        grads_v_loss = [ grad / batch_size for grad in grads_v_loss ] # type: ignore
        grads_w_loss = [ grad / batch_size for grad in grads_w_loss ] # type: ignore

        gradients["v_loss"] = grads_v_loss
        gradients["w_loss"] = grads_w_loss

        if self.u is not None:
            grads_u = tape.gradient(u_loss, [self.u])
            grads_u = [ grad / batch_size for grad in grads_u ] # type: ignore
            gradients["u_loss"] = grads_u

        # ---- #

        del tape

        return values, loss, gradients

    def evaluate_step(
            self,
            obs_init, obs, act, obs_next,
            probs_init, probs_next,
            batch_size):

        values, loss, gradients = self.get_gradients(
            obs_init, obs, act, obs_next,
            probs_init, probs_next,
            batch_size, ) # type: ignore

        self.v_optimizer.apply_gradients(
            zip(gradients["v_loss"], self.v.variables), )

        self.w_optimizer.apply_gradients(
            zip(gradients["w_loss"], self.w.variables), )

        if "u_loss" in gradients.keys():
            self.u_optimizer.apply_gradients( # type: ignore
                zip(gradients["u_loss"], [self.u]), )

        return values, loss, gradients

    def evaluate_loop(self, n_steps, verbosity=1, pbar_keys=None):
        self.i_step_tf = tf.Variable(0, dtype=tf.int64)
        tf.summary.experimental.set_step(self.i_step_tf)

        pbar = range(n_steps)
        if verbosity > 0:
            print(self.id)
            pbar = tqdm(pbar)

        for i_step in pbar:
            samples = self.get_sample()

            values, loss, gradients = self.evaluate_step(
                *samples, self.batch_size, )

            mean_loss = np.nanmean(loss)
            if np.isnan(mean_loss): break

            if i_step % self.save_interval == 0 or i_step == n_steps - 1:
                d = {}

                pv_s = self.solve_pv(weighted=False)
                pv_w = self.solve_pv(weighted=True)

                tf.summary.scalar("loss", mean_loss)
                tf.summary.scalar("pv_s", pv_s)
                tf.summary.scalar("pv_w", pv_w)

                d["loss"] = float(mean_loss)
                d["pv_s"] = float(pv_s) # type: ignore
                d["pv_w"] = float(pv_w) # type: ignore

                recordings = self.get_recordings(
                    self,
                    *samples,
                    values, loss, gradients,
                    pv_s, pv_w, )

                for k, v in recordings.items():
                    tf.summary.scalar(k, v)
                    if pbar_keys is None or k in pbar_keys: d[k] = v

                if verbosity > 0: pbar.set_postfix(d) # type: ignore

            self.i_step_tf.assign_add(1)

    def get_sample(self):
        sample = self.dataset.sample(n=self.batch_size, replace=True)

        obs_init = self.preprocess_obs(sample["obs_init"])
        obs      = self.preprocess_obs(sample["obs"])
        act      = self.preprocess_act(sample["act"])
        obs_next = self.preprocess_obs(sample["obs_next"])

        probs_init = self.preprocess_probs(sample["probs_init"])
        probs_next = self.preprocess_probs(sample["probs_next"])

        return obs_init, obs, act, obs_next, probs_init, probs_next

    def solve_pv(self, weighted):

        obs = self.preprocess_obs(self.dataset["obs"])
        act = self.preprocess_act(self.dataset["act"])
        rew = self.preprocess_rew(self.dataset["rew"])

        sdc = self.get_value(self.w, obs, act)

        a = tf.reduce_sum(sdc * rew)
        b = tf.reduce_sum(sdc) if weighted else len(sdc)
        pv = a / b

        return pv

# ---------------------------------------------------------------- #
