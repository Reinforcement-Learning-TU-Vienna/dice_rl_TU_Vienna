# ---------------------------------------------------------------- #

import tensorflow as tf

from tensorflow.keras.layers import Dense # type: ignore
from tf_agents.policies import tf_policy
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from dice_rl.data.dataset import convert_to_tfagents_timestep, EnvStep
from dice_rl.utils.common import is_categorical_spec, reverse_broadcast

from dice_rl_TU_Vienna.sample import get_probs, get_probs_log

# ---------------------------------------------------------------- #

class NeuralDice(object):

    def __init__(
            self,
            dataset_spec,
            v,
            w,
            v_optimizer,
            w_optimizer,
            gamma: Union[float, tf.Tensor],
            reward_fn: Optional[Callable] = None,
            obs_act: bool = True,
            num_samples: Optional[int] = None,
            v_regularizer: float = 0.0,
            w_regularizer: float = 0.0,
        ):

        self.dataset_spec = dataset_spec

        self.v = v
        self.v.create_variables()

        self.w = w
        self.w.create_variables()

        self.u = None

        self.v_optimizer = v_optimizer
        self.w_optimizer = w_optimizer

        self.v_regularizer = v_regularizer
        self.w_regularizer = w_regularizer

        self.gamma = gamma
        self.reward_fn = reward_fn if reward_fn is None \
            else lambda env_step: env_step.reward
        self.num_samples = num_samples

        self.obs_act = obs_act
        A = not self.obs_act
        B = not self.dataset_spec.has_log_probability()
        if A and B: raise ValueError(
            'Dataset must contain log-probability when obs_act is False.')

        self.categorical_action = is_categorical_spec(
            self.dataset_spec.action)
        A = not self.categorical_action
        B = self.num_samples is None
        if A and B: self.num_samples = 1

        self.v_regularizer = v_regularizer
        self.w_regularizer   = w_regularizer

    def get_value(self, network, env_step):
        x = env_step.observation, env_step.action
        y = env_step.observation,
        A = self.obs_act
        inputs = x if A else y

        value, _ = network(inputs)
        return value

    def get_average_value(self, network, env_step, policy=None):
        if self.obs_act:

            if self.categorical_action and self.num_samples is None:
                action_weights = get_probs(env_step, policy)

                action_dtype = self.dataset_spec.action.dtype
                A = tf.shape(action_weights)
                batch_size  = A[0] # type: ignore
                num_actions = A[-1] # type: ignore

                A = tf.ones([batch_size, 1], dtype=action_dtype)
                B = tf.range(num_actions, dtype=action_dtype)
                B = B[None, :]
                actions = A * B

            else:
                batch_size = tf.shape(env_step.observation)[0] # type: ignore
                num_actions = self.num_samples
                action_weights = tf.ones([batch_size, num_actions]) / num_actions

                tfagents_step = convert_to_tfagents_timestep(env_step)
                A = [policy.action(tfagents_step).action for _ in range(num_actions)] # type: ignore
                actions = tf.stack(A, axis=1)

            tensor = actions
            shape = [batch_size * num_actions] + actions.shape[2:].as_list() # type: ignore
            flat_actions = tf.reshape(tensor, shape)

            input = env_step.observation[:, None, ...]
            multiples = [1, num_actions] + [1] * len(env_step.observation.shape[1:] )
            tensor = tf.tile(input, multiples)
            shape = [batch_size * num_actions] + env_step.observation.shape[1:].as_list()
            flat_observations = tf.reshape(tensor, shape)

            flat_values, _ = network( (flat_observations, flat_actions) )

            tensor = flat_values
            shape = [batch_size, num_actions] + flat_values.shape[1:].as_list()
            values = tf.reshape(flat_values, shape)

            input_tensor = values * reverse_broadcast(action_weights, values)
            return tf.reduce_sum(input_tensor, axis=1)

        else:
            values, _ = network( (env_step.observation,) )
            return values

    def orthogonal_regularization(self, network):
        regularizer = 0
        for layer in network.layers:
            if isinstance(layer, Dense):
                prod = tf.matmul(
                    tf.transpose(layer.kernel), layer.kernel )
                regularizer += tf.reduce_sum(
                    tf.math.square(prod * (1 - tf.eye(prod.shape[0]))) )
        return regularizer

    def get_values(
            self,
            env_step_init, env_step, env_step_next,
            policy=None):

        v_init = self.get_average_value(self.v, env_step_init, policy)
        v      = self.get_value        (self.v, env_step)
        v_next = self.get_average_value(self.v, env_step_next, policy)
        w      = self.get_value        (self.w, env_step)

        policy_ratio = 1.0
        if not self.obs_act:
            A = get_probs_log(env_step, policy)
            B = env_step.get_log_probability()
            policy_ratio = tf.exp(A - B)

        discounts_policy_ratio = reverse_broadcast(
            self.gamma * policy_ratio, v) # type: ignore

        return v_init, v, v_next, w, discounts_policy_ratio

    def get_loss(
            self,
            v_init, v, v_next, w,
            discounts_policy_ratio):

        raise NotImplementedError

    def get_gradients(
            self,
            env_step_init: EnvStep, env_step: EnvStep, env_step_next: EnvStep,
            batch_size: int,
            target_policy: Union[tf_policy.TFPolicy, None] = None):

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch([variable.value for variable in self.v.variables])
            tape.watch([variable.value for variable in self.w.variables])
            if self.u is not None:
                tape.watch([self.u])

            values = self.get_values(env_step_init, env_step, env_step_next, target_policy)
            loss = self.get_loss(*values)

            v_loss = -loss
            w_loss =  loss
            u_loss = -loss

            A = self.v_regularizer
            B = self.orthogonal_regularization(self.v)
            v_loss += A * B

            A = self.w_regularizer
            B = self.orthogonal_regularization(self.w)
            w_loss += A * B

        gradients = {}

        # ---- #
        # values

        v_init, v, v_next, w, _ = values

        grads_v_init = tape.gradient(v_init, self.v.variables)
        grads_v      = tape.gradient(v,      self.v.variables)
        grads_v_next = tape.gradient(v_next, self.v.variables)
        grads_w      = tape.gradient(w,      self.w.variables)

        grads_v_init = [grad / batch_size for grad in grads_v_init] # type: ignore
        grads_v      = [grad / batch_size for grad in grads_v]      # type: ignore
        grads_v_next = [grad / batch_size for grad in grads_v_next] # type: ignore
        grads_w      = [grad / batch_size for grad in grads_w]      # type: ignore

        gradients["v_init"] = grads_v_init
        gradients["v"]      = grads_v
        gradients["v_next"] = grads_v_next
        gradients["w"]      = grads_w

        # ---- #
        # loss

        grads_v_loss = tape.gradient(v_loss, self.v.variables)
        grads_w_loss = tape.gradient(w_loss, self.w.variables)

        grads_v_loss = [grad / batch_size for grad in grads_v_loss] # type: ignore
        grads_w_loss = [grad / batch_size for grad in grads_w_loss] # type: ignore

        gradients["v_loss"] = grads_v_loss
        gradients["w_loss"]   = grads_w_loss

        if self.u is not None:
            grads_u = tape.gradient(u_loss, [self.u])
            grads_u = [grad / batch_size for grad in grads_u] # type: ignore
            gradients["u_loss"] = grads_u

        # ---- #

        return values, loss, gradients

    def eval_step(
            self,
            env_step_init: EnvStep, env_step: EnvStep, env_step_next: EnvStep,
            batch_size: int,
            target_policy: Union[tf_policy.TFPolicy, None] = None):

        values, loss, gradients = self.get_gradients(
            env_step_init, env_step, env_step_next, batch_size, target_policy, )

        self.v_optimizer.apply_gradients(
            zip(gradients["v_loss"], self.v.variables), )

        self.w_optimizer.apply_gradients(
            zip(gradients["w_loss"], self.w.variables), )

        if "u_loss" in gradients.keys():
            self.u_optimizer.apply_gradients( # type: ignore
                zip(gradients["u_loss"], [self.u]), )

        return values, loss, gradients

# ---------------------------------------------------------------- #
