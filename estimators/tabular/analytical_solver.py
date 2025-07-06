# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

from dice_rl_TU_Vienna.estimators.tabular.bellman_equations import (
    solve_forward_bellman_equations, solve_backward_bellman_equations, )

from dice_rl_TU_Vienna.utils.numpy import safe_divide

# ---------------------------------------------------------------- #

class AnalyticalSolver(ABC):
    @property
    def __name__(self): return "analytical"

    def __init__(self, n_obs, n_act):
        self.n_obs = n_obs
        self.n_act = n_act
        self.dim = self.n_obs * self.n_act

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

    def solve_avf(self, gamma, **kwargs):
        projected = kwargs.get("projected", False)

        return solve_forward_bellman_equations(
            dD=self.dD, r=self.r, P=self.P, gamma=gamma, projected=projected)

    def solve_sdc(self, gamma, **kwargs):
        projected = kwargs.get("projected", False)

        return solve_backward_bellman_equations(
            d0=self.d0, dD=self.dD, P=self.P, gamma=gamma, projected=projected)

    def solve_pv_primal(self, gamma, Q):
        return (1 - gamma) * np.dot(self.d0, Q)

    def solve_pv_dual(self, sd):
        return np.dot(sd, self.r)
    
    def solve(self, gamma, **kwargs):

        primal_dual        = kwargs.get("primal_dual", "dual")
        pv_max_duality_gap = kwargs.get("pv_max_duality_gap", 1e-10)

        if primal_dual == "both":
            kwargs_p = kwargs.copy(); kwargs_p["primal_dual"] = "primal"
            kwargs_d = kwargs.copy(); kwargs_d["primal_dual"] = "dual"

            rho_p, info_p = self.solve(gamma, **kwargs_p) # type: ignore
            rho_d, info_d = self.solve(gamma, **kwargs_d) # type: ignore

            assert np.abs(rho_p - rho_d) < pv_max_duality_gap

            rho = np.random.choice([rho_p, rho_d])
            info = { **info_p, **info_d}

            return rho, info

        elif primal_dual == "primal":
            Q, info = self.solve_avf(gamma, **kwargs)
            rho = self.solve_pv_primal(gamma, Q)

            info["Q"] = Q

            return rho, info

        elif primal_dual == "dual":
            w, info = self.solve_sdc(gamma, **kwargs)
            d = w * self.dD

            rho = self.solve_pv_dual(d)

            info["d"] = d
            info["w"] = w

            return rho, info

        else:
            return NotImplementedError

    def errors(
            self,
            gamma,
            pv_approx=None, vf_approx=None, sdc_approx=None,
            vf_approx_network=None, sdc_approx_network=None,
            pv_exact=None, vf_exact=None, sdc_exact=None):

        self.assert_gamma(gamma)

        if None in [pv_exact, sdc_exact, vf_exact]:
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
            np.identity(self.n_obs)[np.repeat(np.arange(self.n_obs), 2)],
            dtype=tf.float32)
        act = tf.convert_to_tensor(
            np.tile(np.arange(self.n_act), self.n_obs),
            dtype=tf.int64)
        vector, _ = network( (obs, act) )

        return vector

# ---------------------------------------------------------------- #
