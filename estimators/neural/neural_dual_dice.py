# ---------------------------------------------------------------- #

import tensorflow as tf

from dice_rl_TU_Vienna.estimators.neural.neural_dice import NeuralDice

# ---------------------------------------------------------------- #

class NeuralDualDice(NeuralDice):
    @property
    def __name__(self): return "NeuralDualDice"

    def get_loss(self, v_init, v, v_next, w):

        g = self.gamma

        x = (1 - g) * v_init
        y = w * ( g * v_next - v )
        z = self.q_fn(w)

        loss = x + y + z

        return loss

    def __init__(
            self,
            gamma, p, seed, batch_size,
            learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs=None, preprocess_act=None, preprocess_rew=None,
            dir=None, get_recordings=None, other_hyperparameters=None, save_interval=100):
    
        if other_hyperparameters is None: other_hyperparameters = {}
        other_hyperparameters["p"] = p

        super().__init__(
            gamma, seed, batch_size,
            learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings, other_hyperparameters, save_interval,
        )

        assert p > 1

        q = p / (p - 1)

        self.p_fn = lambda x: tf.abs(x) ** p / p
        self.q_fn = lambda x: tf.abs(x) ** q / q

# ---------------------------------------------------------------- #
