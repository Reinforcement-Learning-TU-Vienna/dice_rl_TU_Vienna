# ---------------------------------------------------------------- #

import tensorflow as tf

from typing import Callable, Optional, Union

from dice_rl_TU_Vienna.estimators.neural.neural_dice import NeuralDice

# ---------------------------------------------------------------- #

class NeuralDualDice(NeuralDice):
    def get_loss(
            self,
            v_init, v, v_next,
            w,
            discounts_policy_ratio):

        g = self.gamma
        g_prime = discounts_policy_ratio
        v_0 = v_init
        v = v
        v_prime = v_next
        w = w

        x = (1 - g) * v_0 # type: ignore
        y = w * ( g_prime * v_prime - v )
        z = self.fstar_fn(w)

        loss = x + y + z

        return loss

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
            f_exponent: float = 1.5,
        ):

        super().__init__(
            dataset_spec,
            v, w,
            v_optimizer, w_optimizer,
            gamma, reward_fn,
            obs_act, num_samples,
            v_regularizer, w_regularizer,
        )

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1.')
        fstar_exponent = f_exponent / (f_exponent - 1)

        self.f_fn     = lambda x: tf.abs(x) ** f_exponent     / f_exponent
        self.fstar_fn = lambda x: tf.abs(x) ** fstar_exponent / fstar_exponent

# ---------------------------------------------------------------- #
