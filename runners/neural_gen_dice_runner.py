# ---------------------------------------------------------------- #

import tensorflow as tf

from tensorflow.keras.optimizers import SGD # type: ignore

from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice import NeuralGenDice
from dice_rl_TU_Vienna.runners.neural_dice_runner import NeuralDiceRunner, lr_to_str

# ---------------------------------------------------------------- #

class NeuralGenDiceRunner(NeuralDiceRunner):
    @property
    def __name__(self): return "NeuralGenDice"

    def __init__(
        self,
        gamma, num_steps, batch_size, seed,
        v_hidden_dims, w_hidden_dims,
        v_learning_rate, w_learning_rate,
        v_regularizer, w_regularizer,
        u_learning_rate, lam, # new
        dataset, dataset_spec=None, target_policy=None,
        save_dir=None,
        by="steps", analytical_solver=None,
        env_step_preprocessing=None, aux_recorder=None, aux_recorder_pbar=None,
        verbosity=1):

        self.u_learning_rate = u_learning_rate
        self.lam = lam

        super().__init__(
            gamma, num_steps, batch_size, seed,
            v_hidden_dims, w_hidden_dims,
            v_learning_rate, w_learning_rate,
            v_regularizer, w_regularizer,
            dataset, dataset_spec, target_policy,
            save_dir,
            by, analytical_solver,
            env_step_preprocessing, aux_recorder, aux_recorder_pbar,
            verbosity,
        )

    @property
    def hparam_str_evaluation(self):
        nlr = lr_to_str(self.u_learning_rate)

        return "_".join([
            super().hparam_str_evaluation,
            f"nlr{nlr}", f"nreg{self.lam}",
        ])

    def set_up_estimator(self):
        v, w, v_optimizer, w_optimizer = super().set_up_estimator()

        u_optimizer = SGD(self.u_learning_rate)

        self.estimator = NeuralGenDice(
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
    def dual_output_activation_fn(self): return tf.math.square

# ---------------------------------------------------------------- #
