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
            network_primal,
            network_dual,
            optimizer_primal,
            optimizer_dual,
            gamma: Union[float, tf.Tensor],
            reward_fn: Optional[Callable] = None,
            solve_for_state_action_ratio: bool = True,
            n_samples: Optional[int] = None,
            regularizer_primal: float = 0.0,
            regularizer_dual: float = 0.0,
        ):

        self.dataset_spec = dataset_spec

        self.network_primal = network_primal
        self.network_primal.create_variables()

        self.network_dual = network_dual
        self.network_dual.create_variables()

        self.network_norm = None

        self.optimizer_primal = optimizer_primal
        self.optimizer_dual = optimizer_dual

        self.regularizer_primal = regularizer_primal
        self.regularizer_dual = regularizer_dual

        self.gamma = gamma
        self.reward_fn = reward_fn if reward_fn is None \
            else lambda env_step: env_step.reward
        self.n_samples = n_samples

        self.solve_for_state_action_ratio = solve_for_state_action_ratio
        A = not self.solve_for_state_action_ratio
        B = not self.dataset_spec.has_log_probability()
        if A and B: raise ValueError(
            'Dataset must contain log-probability when solve_for_state_action_ratio is False.')

        self.categorical_action = is_categorical_spec(
            self.dataset_spec.action)
        A = not self.categorical_action
        B = self.n_samples is None
        if A and B: self.n_samples = 1

        self.regularizer_primal = regularizer_primal
        self.regularizer_dual   = regularizer_dual

    def get_value(self, network, env_step):
        x = env_step.observation, env_step.action
        y = env_step.observation,
        A = self.solve_for_state_action_ratio
        inputs = x if A else y

        value, _ = network(inputs)
        return value

    def get_average_value(self, network, env_step, policy=None):
        if self.solve_for_state_action_ratio:

            if self.categorical_action and self.n_samples is None:
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
                num_actions = self.n_samples
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

        primal_values_init = self.get_average_value(
            self.network_primal, env_step_init, policy)

        primal_values = self.get_value(
            self.network_primal, env_step)

        primal_values_next = self.get_average_value(
            self.network_primal, env_step_next, policy)

        dual_values = self.get_value(
            self.network_dual, env_step)

        discounts = self.gamma # * env_step_next.discount

        policy_ratio = 1.0
        if not self.solve_for_state_action_ratio:
            A = get_probs_log(env_step, policy)
            B = env_step.get_log_probability()
            policy_ratio = tf.exp(A - B)

        discounts_policy_ratio = reverse_broadcast(
            discounts * policy_ratio, primal_values) # type: ignore

        return (
            primal_values_init, primal_values, primal_values_next,
            dual_values,
            discounts_policy_ratio,
        )

    def get_loss(
            self,
            primal_values_init, primal_values, primal_values_next,
            dual_values,
            discounts_policy_ratio):

        raise NotImplementedError

    def get_gradients(
            self,
            env_step_init: EnvStep, env_step: EnvStep, env_step_next: EnvStep,
            batch_size: int,
            target_policy: Union[tf_policy.TFPolicy, None] = None):

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch([variable.value for variable in self.network_primal.variables])
            tape.watch([variable.value for variable in self.network_dual  .variables])
            if self.network_norm is not None:
                tape.watch([self.network_norm])

            values = self.get_values(env_step_init, env_step, env_step_next, target_policy)
            loss = self.get_loss(*values)

            loss_primal = -loss
            loss_dual   =  loss
            loss_norm   = -loss

            A = self.regularizer_primal
            B = self.orthogonal_regularization(self.network_primal)
            loss_primal += A * B

            A = self.regularizer_dual
            B = self.orthogonal_regularization(self.network_dual)
            loss_dual += A * B

        gradients = {}

        grads_primal = tape.gradient(loss_primal, self.network_primal.variables)
        grads_dual   = tape.gradient(loss_dual,   self.network_dual  .variables)

        grads_primal = [grad / batch_size for grad in grads_primal] # type: ignore
        grads_dual   = [grad / batch_size for grad in grads_dual  ] # type: ignore

        gradients["primal"] = grads_primal
        gradients["dual"]   = grads_dual

        if self.network_norm is not None:
            grads_norm = tape.gradient(loss_norm, [self.network_norm])
            grads_norm = [grad / batch_size for grad in grads_norm]   # type: ignore
            gradients["norm"] = grads_norm

        return gradients, loss

    def eval_step(
            self,
            env_step_init: EnvStep, env_step: EnvStep, env_step_next: EnvStep,
            batch_size: int,
            target_policy: Union[tf_policy.TFPolicy, None] = None):

        gradients, loss = self.get_gradients(
            env_step_init, env_step, env_step_next, batch_size, target_policy, )

        self.optimizer_primal.apply_gradients(
            zip(gradients["primal"], self.network_primal.variables), )

        self.optimizer_dual.apply_gradients(
            zip(gradients["dual"], self.network_dual.variables), )

        if "norm" in gradients.keys():
            self.optimizer_norm.apply_gradients( # type: ignore
                zip(gradients["norm"], [self.network_norm]), )

        return gradients, loss

# ---------------------------------------------------------------- #
