# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice import NeuralGenDice

# ---------------------------------------------------------------- #

class NeuralGradientDice(NeuralGenDice):
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

        loss = x + y + z - 1/2 * v**2

        return loss

# ---------------------------------------------------------------- #
