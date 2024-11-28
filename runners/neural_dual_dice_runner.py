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
        primal_hidden_dims, dual_hidden_dims,
        primal_learning_rate, dual_learning_rate,
        regularizer_primal, regularizer_dual,
        f_exponent, # new
        dataset, dataset_spec=None, target_policy=None,
        save_dir=None,
        by="steps", analytical_solver=None,
        env_step_preprocessing=None, aux_recorder=None, aux_recorder_pbar=None,
        verbosity=1):

        self.f_exponent = f_exponent

        super().__init__(
            gamma, num_steps, batch_size, seed,
            primal_hidden_dims, dual_hidden_dims,
            primal_learning_rate, dual_learning_rate,
            regularizer_primal, regularizer_dual,
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
            network_primal=pn,
            network_dual=dn,
            optimizer_primal=po,
            optimizer_dual=do,
            gamma=self.gamma,
            f_exponent=self.f_exponent,
            regularizer_primal=self.regularizer_primal,
            regularizer_dual=self.regularizer_dual,
        )
 
 # ---------------------------------------------------------------- #
