# ---------------------------------------------------------------- #

import tensorflow as tf

from dice_rl_TU_Vienna.estimators.neural.neural_gradient_dice import NeuralGradientDice
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner import NeuralGenDiceRunner

# ---------------------------------------------------------------- #

class NeuralGradientDiceRunner(NeuralGenDiceRunner):
    @property
    def __name__(self): return "NeuralGradientDice"

    def set_up_estimator(self):
        pn, dn, po, do, no = super().set_up_estimator()

        self.estimator = NeuralGradientDice(
            self.dataset_spec,
            network_primal=pn,
            network_dual=dn,
            optimizer_primal=po,
            optimizer_dual=do,
            optimizer_norm=no,
            gamma=self.gamma,
            regularizer_primal=self.regularizer_primal,
            regularizer_dual=self.regularizer_dual,
            regularizer_norm=self.regularizer_norm,
        )

        return pn, dn, po, do, no
    
    @property
    def dual_output_activation_fn(self): return tf.identity

# ---------------------------------------------------------------- #
