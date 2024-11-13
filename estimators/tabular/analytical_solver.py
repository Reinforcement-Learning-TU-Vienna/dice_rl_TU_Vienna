# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

from dice_rl_TU_Vienna.estimators.tabular.bellman_equations import (
    solve_forwards_bellman_equations, solve_backwards_bellman_equations, )

from utils.numpy import safe_divide

# ---------------------------------------------------------------- #

class AnalyticalSolver(ABC):
    @property
    def __name__(self): return "analytical"

    def __init__(self, num_obs, n_act):
        self.num_obs = num_obs
        self.n_act = n_act
        self.dim = self.num_obs * self.n_act

        self.d0, self.dD, self.P, self.r = self.get_distributions() # type: ignore

    def assert_gamma(self, gamma):
        pass    

    def get_index(self, obs, act=None):
        if act is None:
            i1 = obs * self.n_act
            i2 = i1 + self.n_act
            return i1, i2

        else:
            i = obs * self.n_act + act
            return i

    @abstractmethod
    def get_distributions(self):
        pass

    def get_avf(self, gamma, projected=False):
        return solve_forwards_bellman_equations(
            dD=self.dD, r=self.r, P=self.P, gamma=gamma, projected=projected)

    def get_sdc(self, gamma, projected=False):
        return solve_backwards_bellman_equations(
            d0=self.d0, dD=self.dD, P=self.P, gamma=gamma, projected=projected)

    def get_pv_primal(self, gamma, Q):
        return (1 - gamma) * np.dot(self.d0, Q)

    def get_pv_dual(self, sd):
        return np.dot(sd, self.r)
    
    def solve(self, gamma, primal_dual="both", projected=False, pv_max_duality_gap=1e-10):
        if primal_dual == "both":
            pv_p, Q,   info_p = self.solve(gamma, primal_dual="primal", projected=projected) # type: ignore
            pv_d, sdc, info_d = self.solve(gamma, primal_dual="dual",   projected=projected) # type: ignore

            assert np.abs(pv_p - pv_d) < pv_max_duality_gap

            pv = np.random.choice([pv_p, pv_d])
            info = { **info_p, **info_d}

            return pv, (Q, sdc), info

        elif primal_dual == "primal":
            Q, info = self.get_avf(gamma, projected=projected)
            pv = self.get_pv_primal(gamma, Q)
            return pv, Q, info

        elif primal_dual == "dual":
            sdc, info = self.get_sdc(gamma, projected=projected)
            sd = sdc * self.dD
            pv = self.get_pv_dual(sd)
            return pv, sdc, info

        else:
            return NotImplementedError

    def errors(
            self,
            gamma,
            pv_approx=None, vf_approx=None, sdc_approx=None,
            vf_approx_network=None, sdc_approx_network=None,
            pv_exact=None, vf_exact=None, sdc_exact=None,
            solve_if_None=False):

        self.assert_gamma(gamma)

        if solve_if_None:
            A = pv_exact  is None
            B = sdc_exact is None
            C = vf_exact  is None

            if A or B or C:
                pv_exact, (vf_exact, sdc_exact), _ = self.solve(gamma) # type: ignore

        if sdc_approx_network is not None:
            sdc_approx = self.network_to_vector(sdc_approx_network)

        # ---------------- #

        d0 = self.d0
        dD = self.dD
        P = self.P

        w = sdc_exact
        w_hat = np.array(sdc_approx)

        rho = pv_exact
        rho_hat = pv_approx

        # ---------------- #

        errors = {}

        if rho is not None:
            if isinstance(rho_hat, dict):
                for k, v in rho_hat.items():
                    delta = rho - v
                    errors[f"pv_error_{k}"] = np.abs(delta)
            else:
                delta = rho - rho_hat
                errors[f"pv_error"] = np.abs(delta)

        if w is not None:
            delta = w - w_hat
            errors["sdc_L1_error"] = np.dot(dD, np.abs(delta) ** 1)
            errors["sdc_L2_error"] = np.dot(dD, np.abs(delta) ** 2)

        x = dD * w_hat
        y = (1 - gamma) * d0 + gamma * P.T @ ( dD * w_hat )
        delta = safe_divide(x - y, dD)
        errors["bellman_L1_error"] = np.dot(dD, np.abs(delta) ** 1)
        errors["bellman_L2_error"] = np.dot(dD, np.abs(delta) ** 2)

        errors["norm_error"] = np.abs(np.dot(dD, w_hat) - 1)

        negative = np.sum(np.abs(w_hat[w_hat < 0]))
        total    = np.sum(np.abs(w_hat))
        errors["negativity"] = negative / total if total != 0 else 0

        return errors

    def network_to_vector(self, network):
        obs = tf.convert_to_tensor(
            np.identity(self.num_obs)[np.repeat(np.arange(self.num_obs), 2)],
            dtype=tf.float32)
        act = tf.convert_to_tensor(
            np.tile(np.arange(self.n_act), self.num_obs),
            dtype=tf.int64)
        vector, _ = network( (obs, act) )

        return vector

# ---------------------------------------------------------------- #
