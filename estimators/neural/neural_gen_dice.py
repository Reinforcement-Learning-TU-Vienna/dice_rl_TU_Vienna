# ---------------------------------------------------------------- #

import tensorflow as tf

from tensorflow.keras.optimizers import SGD # type: ignore

from dice_rl_TU_Vienna.estimators.neural.neural_dice import NeuralDice

# ---------------------------------------------------------------- #

class NeuralGenDice(NeuralDice):
    @property
    def __name__(self): return "NeuralGenDice"

    def get_loss(self, v_init, v, v_next, w):

        g = self.gamma
        l = self.lamda
        u = self.u

        x = (1 - g) * v_init
        y = w * ( g * v_next - v )
        z = l * ( u * (w - 1) - 1/2 * u**2 ) # type: ignore

        loss = x + y + z - 1/4 * v**2 * w

        return loss

    def __init__(
            self,
            gamma, lamda, seed, batch_size,
            learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs=None, preprocess_act=None, preprocess_rew=None,
            dir=None, get_recordings=None, other_hyperparameters=None, save_interval=100):

        super().__init__(
            gamma, seed, batch_size,
            learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings, other_hyperparameters, save_interval,
        )

        self.lamda = lamda

        self.hyperparameters["lamda"] = lamda

    def set_up_networks(self):
        super().set_up_networks()

        self.u = tf.Variable(1.0)
        self.u_optimizer = SGD(self.learning_rate)

# ---------------------------------------------------------------- #
