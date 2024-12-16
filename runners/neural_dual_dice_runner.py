# ---------------------------------------------------------------- #

import tensorflow as tf

from dice_rl_TU_Vienna.estimators.neural.neural_dual_dice import NeuralDualDice
from dice_rl_TU_Vienna.runners.neural_dice_runner import NeuralDiceRunner

# ---------------------------------------------------------------- #

class NeuralDualDiceRunner(NeuralDiceRunner):
    @property
    def __name__(self): return "NeuralDualDice"

    def __init__(
        self,
        gamma, num_steps, batch_size, seed,
        v_hidden_dims, w_hidden_dims,
        v_learning_rate, w_learning_rate,
        v_regularizer, w_regularizer,
        f_exponent, # new
        dataset, dataset_spec=None, target_policy=None,
        save_dir=None,
        by="steps", analytical_solver=None,
        env_step_preprocessing=None, aux_recorder=None, aux_recorder_pbar=None,
        verbosity=1):

        self.f_exponent = f_exponent

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
        return "_".join([
            super().hparam_str_evaluation,
            f"fexp{self.f_exponent}",
        ])

    def set_up_estimator(self):
        pn, dn, po, do = super().set_up_estimator()
        
        self.estimator = NeuralDualDice(
            self.dataset_spec,
            v=pn,
            w=dn,
            v_optimizer=po,
            w_optimizer=do,
            gamma=self.gamma,
            f_exponent=self.f_exponent,
            v_regularizer=self.v_regularizer,
            w_regularizer=self.w_regularizer,
        )
 
 # ---------------------------------------------------------------- #
