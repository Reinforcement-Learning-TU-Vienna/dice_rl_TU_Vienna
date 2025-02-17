# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice import NeuralGenDice

# ---------------------------------------------------------------- #

class NeuralGradientDice(NeuralGenDice):
    @property
    def __name__(self): return "NeuralGradientDice"

    def get_loss(self, v_init, v, v_next, w):

        g = self.gamma
        l = self.lamda
        u = self.u

        x = (1 - g) * v_init
        y = w * ( g * v_next - v )
        z = l * ( u * (w - 1) - 1/2 * u**2 ) # type: ignore

        loss = x + y + z - 1/2 * v**2

        return loss

# ---------------------------------------------------------------- #
