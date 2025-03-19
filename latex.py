# ---------------------------------------------------------------- #

latex_gamma = r"\gamma"
latex_lambda = r"\lambda"

latex_rho_pi   = r"\rho^\pi"
latex_rho_pi_s = r"\rho^\pi_\text{S}"
latex_rho_pi_w = r"\rho^\pi_\text{W}"
latex_rho_pi_hat   = r"\hat{\rho}^\pi"
latex_rho_pi_hat_s = r"\hat{\rho}^\pi_\text{S}"
latex_rho_pi_hat_w = r"\hat{\rho}^\pi_\text{W}"

latex_w_pi_D = r"w_{\pi / D}"
latex_w_pi_D_hat = r"\hat{w}_{\pi / D}"

latex_div = r"\div"

latex_E = r"\mathbb{E}_D"
latex_E_hat = r"\hat{\mathbb {E}}_D"

latex_d_0 = r"d_0^\pi"
latex_d_0_hat = r"\hat{d}_0^\pi"

latex_P = r"P"
latex_P_hat = r"\hat{P}"

latex_P_star = r"P_\ast"
latex_P_star_hat = r"\hat{P}_\ast"

latex_D = r"D"
latex_D_hat = r"\hat{D}"

latex_D_inv = r"D^{-1}"
latex_D_inv_hat = r"\hat{D}^{-1}"

latex_cdot = r"\cdot"

latex_J = r"J"
latex_J_hat = r"\hat{J}"

# ---------------------------------------------------------------- #

def latex_pv(gamma=None):
    if gamma == "":     s = f"{latex_rho_pi}"
    elif gamma is None: s = f"{latex_rho_pi}({latex_gamma})"
    else:               s = f"{latex_rho_pi}({latex_gamma} = {gamma})"
    return s

def latex_pv_approx(gamma=None, weighted=None):

    if weighted is None:
        x = latex_rho_pi_hat
    else:
        if weighted: x = latex_rho_pi_hat_w
        else:        x = latex_rho_pi_hat_s

    if gamma == "":     s = f"{x}"
    elif gamma is None: s = f"{x}({latex_gamma})"
    else:               s = f"{x}({latex_gamma} = {gamma})"
 
    return s

def latex_pv_approx_s(gamma=None):
    return latex_pv_approx(gamma, False)

def latex_pv_approx_w(gamma=None):
    return latex_pv_approx(gamma, True)

def latex_sdc(gamma=None):
    if gamma == "":     s = f"{latex_w_pi_D}"
    elif gamma is None: s = f"{latex_w_pi_D}({latex_gamma})"
    else:               s = f"{latex_w_pi_D}({latex_gamma} = {gamma})"
    return s

def latex_sdc_approx(gamma=None):
    if gamma == "":     s = f"{latex_w_pi_D_hat}"
    elif gamma is None: s = f"{latex_w_pi_D_hat}({latex_gamma})"
    else:               s = f"{latex_w_pi_D_hat}({latex_gamma} = {gamma})"
    return s


def latex_pv_scaled(gamma=None):
    if gamma == "":     s = f"{latex_rho_pi} {latex_div} (1 - {latex_gamma})"
    elif gamma is None: s = f"{latex_rho_pi}({latex_gamma}) {latex_div} (1 - {latex_gamma})"
    else:               s = f"{latex_rho_pi}({latex_gamma} = {gamma}) {latex_div} {1 - gamma}"
    return s


def latex_pv_error(gamma=None, weighted=None):
    x = latex_pv_approx(gamma, weighted)
    y = latex_pv       (gamma)
    s = f"| {x} - {y} |"
    return s

def latex_pv_error_s(gamma=None):
    return latex_pv_error(gamma, False)

def latex_pv_error_w(gamma=None):
    return latex_pv_error(gamma, True)

def latex_sdc_Lp_error(p, gamma=None):
    x = latex_sdc_approx(gamma)
    y = latex_sdc       (gamma)
    s = f"{latex_E}| {x} - {y} |^{p}"
    return s

def latex_sdc_L1_error(gamma=None): return latex_sdc_Lp_error(1, gamma)
def latex_sdc_L2_error(gamma=None): return latex_sdc_Lp_error(2, gamma)

def latex_bellman_Lp_error(p, gamma=None):
    x = latex_sdc_approx(gamma)
    a = f"(1 - {latex_gamma}) {latex_cdot} {latex_D_inv} {latex_d_0}"
    b = f"{latex_gamma} {latex_cdot} {latex_D_inv} {latex_P_star} {latex_D} {x}"
    c = f"{latex_D} {x}"
    s = f"{latex_E} | {a} + {b} - {c} |^{p}"
    return s

def latex_bellman_L1_error(gamma=None): return latex_bellman_Lp_error(1, gamma)
def latex_bellman_L2_error(gamma=None): return latex_bellman_Lp_error(2, gamma)

def latex_norm_error(gamma=None):
    x = latex_sdc_approx(gamma)
    s = f"| {latex_E}[{x}] - 1 |"
    return s


def latex_loss(gamma=None):
    x = latex_sdc_approx(gamma)
    s = f"{latex_J_hat}(v, {x}, u)"
    return s

# ---------------------------------------------------------------- #

def dollar(f):

    def g(*args, **kwargs):
        s = f(*args, *kwargs)
        return f"${s}$"

    return g

latex_labels = {
    "pv":  dollar(latex_pv),
    "sdc": dollar(latex_sdc),
    "pv_approx":  dollar(latex_pv_approx),
    "pv_approx_s":  dollar(latex_pv_approx_s),
    "pv_approx_w":  dollar(latex_pv_approx_w),
    "sdc_approx": dollar(latex_sdc_approx),
    #
    "pv_scaled": dollar(latex_pv_scaled),
    #
    "pv_error":         dollar(latex_pv_error),
    "pv_error_s":       dollar(latex_pv_error_s),
    "pv_error_w":       dollar(latex_pv_error_w),
    "sdc_L1_error":     dollar(latex_sdc_L1_error),
    "sdc_L2_error":     dollar(latex_sdc_L2_error),
    "bellman_L1_error": dollar(latex_bellman_L1_error),
    "bellman_L2_error": dollar(latex_bellman_L2_error),
    "norm_error":       dollar(latex_norm_error),
    #
    "loss": dollar(latex_loss),
}

# ---------------------------------------------------------------- #
