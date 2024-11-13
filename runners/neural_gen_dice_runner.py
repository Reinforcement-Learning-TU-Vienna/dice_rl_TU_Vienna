# ---------------------------------------------------------------- #

import tensorflow as tf

from tensorflow.keras.optimizers import Adam # type: ignore

from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice import NeuralGenDice
from dice_rl_TU_Vienna.runners.neural_dice_runner import NeuralDiceRunner, lr_to_str

# ---------------------------------------------------------------- #

class NeuralGenDiceRunner(NeuralDiceRunner):
    @property
    def __name__(self): return "NeuralGenDice"

    def __init__(
        self,
        gamma, num_steps, batch_size, seed,
        primal_hidden_dims, dual_hidden_dims,
        primal_learning_rate, dual_learning_rate,
        regularizer_primal, regularizer_dual,
        norm_learning_rate, regularizer_norm, # new
        dataset, dataset_spec=None, target_policy=None,
        save_dir=None,
        by="steps", analytical_solver=None,
        env_step_preprocessing=None,
        verbosity=1):

        self.norm_learning_rate = norm_learning_rate
        self.regularizer_norm = regularizer_norm

        super().__init__(
            gamma, num_steps, batch_size, seed,
            primal_hidden_dims, dual_hidden_dims,
            primal_learning_rate, dual_learning_rate,
            regularizer_primal, regularizer_dual,
            dataset, dataset_spec, target_policy,
            save_dir,
            by, analytical_solver,
            env_step_preprocessing,
            verbosity,
        )

    @property
    def hparam_str_evaluation(self):
        nlr = lr_to_str(self.norm_learning_rate)

        return "_".join([
            super().hparam_str_evaluation,
            f"nlr{nlr}", f"nreg{self.regularizer_norm}",
        ])

    def set_up_estimator(self):
        pn, dn, po, do = super().set_up_estimator()

        no = Adam(
            self.norm_learning_rate, clipvalue=1.0)

        self.estimator = NeuralGenDice(
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
    def dual_output_activation_fn(self): return tf.math.square

# ---------------------------------------------------------------- #
