# ---------------------------------------------------------------- #

import tensorflow as tf

from dice_rl_TU_Vienna.estimators.neural.neural_gradient_dice import NeuralGradientDice
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner import NeuralGenDiceRunner

# ---------------------------------------------------------------- #

class NeuralGradientDiceRunner(NeuralGenDiceRunner):
    @property
    def __name__(self): return "NeuralGradientDice"

    def set_up_estimator(self):
        v, w, v_optimizer, w_optimizer, u_optimizer = super().set_up_estimator()

        self.estimator = NeuralGradientDice(
            self.dataset_spec,
            v=v,
            w=w,
            v_optimizer=v_optimizer,
            w_optimizer=w_optimizer,
            u_optimizer=u_optimizer,
            gamma=self.gamma,
            v_regularizer=self.v_regularizer,
            w_regularizer=self.w_regularizer,
            lam=self.lam,
        )

        return v, w, v_optimizer, w_optimizer, u_optimizer
    
    @property
    def dual_output_activation_fn(self): return tf.identity

# ---------------------------------------------------------------- #
