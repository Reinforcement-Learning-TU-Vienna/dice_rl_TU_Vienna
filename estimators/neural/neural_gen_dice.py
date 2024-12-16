# ---------------------------------------------------------------- #

import tensorflow as tf

from typing import Callable, Optional, Union

from dice_rl_TU_Vienna.estimators.neural.neural_dice import NeuralDice

# ---------------------------------------------------------------- #

class NeuralGenDice(NeuralDice):
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

        lam = self.lam
        u = self.u

        x = (1 - g) * v_0 # type: ignore
        y = w * ( g_prime * v_prime - v )
        z = lam * ( u * (w - 1) - 1/2 * u**2 ) # type: ignore

        loss = x + y + z - 1/4 * v**2 * w

        return loss

    def __init__(
            self,
            dataset_spec,
            v,
            w,
            v_optimizer,
            w_optimizer,
            u_optimizer,
            gamma: Union[float, tf.Tensor],
            reward_fn: Optional[Callable] = None,
            obs_act: bool = True,
            num_samples: Optional[int] = None,
            v_regularizer: float = 0.0,
            w_regularizer: float = 0.0,
            lam: float = 1.0,
        ):

        super().__init__(
            dataset_spec,
            v, w,
            v_optimizer, w_optimizer,
            gamma, reward_fn,
            obs_act, num_samples,
            v_regularizer, w_regularizer,
        )

        self.u = tf.Variable(1.0)
        self.u_optimizer = u_optimizer
        self.lam = lam

# ---------------------------------------------------------------- #
