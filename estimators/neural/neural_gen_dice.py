# ---------------------------------------------------------------- #

import tensorflow as tf

from typing import Callable, Optional, Union

from dice_rl_TU_Vienna.estimators.neural.neural_dice import NeuralDice

# ---------------------------------------------------------------- #

class NeuralGenDice(NeuralDice):
    def get_loss(
            self,
            initial_primal_values, primal_values, next_primal_values,
            dual_values,
            discounts_policy_ratio):

        g = self.gamma
        g_prime = discounts_policy_ratio
        v_0 = initial_primal_values
        v = primal_values
        v_prime = next_primal_values
        w = dual_values

        lam = self.regularizer_norm
        u = self.network_norm

        x = (1 - g) * v_0 # type: ignore
        y = w * ( g_prime * v_prime - v )
        z = lam * ( u * (w - 1) - 1/2 * u**2 ) # type: ignore

        loss = x + y + z - 1/4 * v**2 * w

        return loss

    def __init__(
            self,
            dataset_spec,
            network_primal,
            network_dual,
            optimizer_primal,
            optimizer_dual,
            optimizer_norm,
            gamma: Union[float, tf.Tensor],
            reward_fn: Optional[Callable] = None,
            solve_for_state_action_ratio: bool = True,
            n_samples: Optional[int] = None,
            regularizer_primal: float = 0.0,
            regularizer_dual: float = 0.0,
            regularizer_norm: float = 1.0,
        ):

        super().__init__(
            dataset_spec,
            network_primal, network_dual,
            optimizer_primal, optimizer_dual,
            gamma, reward_fn,
            solve_for_state_action_ratio, n_samples,
            regularizer_primal, regularizer_dual,
        )

        self.network_norm = tf.Variable(1.0)
        self.optimizer_norm = optimizer_norm
        self.regularizer_norm = regularizer_norm

# ---------------------------------------------------------------- #
