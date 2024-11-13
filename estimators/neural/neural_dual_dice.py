# ---------------------------------------------------------------- #

import tensorflow as tf

from typing import Callable, Optional, Union

from dice_rl_TU_Vienna.estimators.neural.neural_dice import NeuralDice

# ---------------------------------------------------------------- #

class NeuralDualDice(NeuralDice):
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

        x = (1 - g) * v_0 # type: ignore
        y = w * ( g_prime * v_prime - v )
        z = self.fstar_fn(w)

        loss = x + y + z

        return loss

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
            f_exponent: float = 1.5,
        ):

        super().__init__(
            dataset_spec,
            network_primal, network_dual,
            optimizer_primal, optimizer_dual,
            gamma, reward_fn,
            solve_for_state_action_ratio, n_samples,
            regularizer_primal, regularizer_dual,
        )

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1.')
        fstar_exponent = f_exponent / (f_exponent - 1)

        self.f_fn     = lambda x: tf.abs(x) ** f_exponent     / f_exponent
        self.fstar_fn = lambda x: tf.abs(x) ** fstar_exponent / fstar_exponent

# ---------------------------------------------------------------- #
