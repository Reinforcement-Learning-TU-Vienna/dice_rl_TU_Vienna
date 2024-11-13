# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice import NeuralGenDice

# ---------------------------------------------------------------- #

class NeuralGradientDice(NeuralGenDice):
    def get_loss(
            self,
            initial_primal_values, primal_values, next_primal_values,
            dual_values,
            discounts_policy_ratio):

        g = self.gamma
        g_prime = discounts_policy_ratio
        v_0 = initial_primal_values
        v = primal_values
        v_prime = next_primal_values
        w = dual_values

        lam = self.regularizer_norm
        u = self.network_norm

        x = (1 - g) * v_0 # type: ignore
        y = w * ( g_prime * v_prime - v )
        z = lam * ( u * (w - 1) - 1/2 * u**2 ) # type: ignore

        loss = x + y + z - 1/2 * v**2

        return loss

# ---------------------------------------------------------------- #
