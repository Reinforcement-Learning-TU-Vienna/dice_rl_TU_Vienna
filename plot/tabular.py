# ---------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os

from dice_rl_TU_Vienna.estimators.get import get_gammas_log10

from dice_rl_TU_Vienna.utils.general import safe_zip

# ---------------------------------------------------------------- #

def plot_observations(dataset):
    u, c = np.unique(
        dataset.get_all_steps(include_terminal_steps=False).observation,
        return_counts=True,
    )

    plt.figure( figsize=(24, 4), )
    plt.bar(u, c / sum(c))
    plt.yscale("log")
    plt.grid(linestyle=":")
    plt.show()

def plot_observations_difference(dataset_1, dataset_2, ylim=None):
    u_1, c_1 = np.unique(
        dataset_1.get_all_steps(include_terminal_steps=False).observation,
        return_counts=True,
    )
    d_1 = dict( zip(u_1, c_1 / sum(c_1) ) )

    u_2, c_2 = np.unique(
        dataset_2.get_all_steps(include_terminal_steps=False).observation,
        return_counts=True,
    )
    d_2 = dict( zip(u_2, c_2 / sum(c_2) ) )

    K = set( list( d_1.keys() ) + list( d_2.keys() ) )

    d = { k: d_1.get(k, 0) - d_2.get(k, 0) for k in K }
    d = dict( sorted( d.items() ) )

    plt.figure( figsize=(24, 4), )
    plt.bar( d.keys(), d.values() ) # type: ignore
    plt.ylim(ylim)
    plt.grid(linestyle=":")
    plt.show()

# ---------------------------------------------------------------- #

def plot_sr_pv_error(suptitle, success_rates, estimators, dir=None):
    gammas = get_gammas_log10()

    plt.figure()

    x = 1 - gammas
    markers = [".", "v", "^"]
    colors = ["black", "lightgrey", "grey"]

    for z in safe_zip(estimators, success_rates, colors, markers):
        estimator, success_rate, color, marker = z

        y = [ np.abs( success_rate - estimator(gamma, scale=False) ) for gamma in gammas ]
        label = estimator.__name__
        plt.plot(x, y, label=label, color=color, marker=marker)

    plt.plot(x, x, label=r"$O(x)$", color="black", linestyle=":")

    plt.grid(linestyle=":")

    plt.suptitle(suptitle)
    plt.xlabel(r"$1 - \gamma$")
    plt.ylabel(r"$| \hat{\rho}^\pi(\gamma) \div (1 - \gamma) - \sigma |$")
    plt.legend()

    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis()

    if dir is not None:

        if not tf.io.gfile.isdir(dir):
            tf.io.gfile.makedirs(dir)

        file_name = "sr_pv_error"
        save_path = os.path.join(dir, file_name)

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

# -------------------------------- #

def plot_pvs(
        estimators,
        gammas,
        projected, weighted,
        modified, lamda,
        #
        pvs, pv_lims, pv_ref, pv_lim_ref,
        labels,
        colors, colors_lim,
        markers, markers_lim,
        suptitle,
        one_minus_gamma, errors=False,
        xlabel=None, ylabel=None,
        scale_x=False, scale_y=False,
        scale_pv=False,
        ylim=None, legend=True,
        dir=None):

    if errors:
        assert pv_ref is not None
        if pv_lims is not None:
            assert pv_lim_ref is not None

    plt.figure()

    x = gammas if not one_minus_gamma else 1 - gammas

    # -------------------------------- #

    if pv_lims is not None:

        Z = pv_lims, colors_lim, markers_lim
        for z in safe_zip(*Z):
            pv_lim, color, marker = z

            y = pv_lim if not errors else np.abs(pv_lim - pv_lim_ref)

            plt.axhline(y, color=color, marker=marker, linestyle=":")

    # ---------------- #

    if labels is None: labels = [None] * len(pvs)

    Z = estimators, pvs, labels, colors, markers
    for z in safe_zip(*Z):
        estimator, pv, label, color, marker = z

        y1 = pv if not errors else np.abs(pv - pv_ref)
        y2 = 1 / (1 - gammas) if scale_pv else 1
        y = y1 * y2

        if label is None:
            label = estimator.__name__
            if label == "TabularDice":         label += f", {modified=}"
            if label == "TabularGradientDice": label += f", {lamda=}"

        if label == "": label = None

        plt.plot(x, y, label=label, color=color, marker=marker)

    # -------------------------------- #

    if scale_x: plt.xscale("log")
    if scale_y: plt.yscale("log")

    if one_minus_gamma: plt.gca().invert_xaxis()

    if xlabel is None: xlabel = \
        r"$\gamma$" if not one_minus_gamma else r"$1 - \gamma$"
    plt.xlabel(xlabel)

    if ylabel is None:
        x = r"$\rho^\pi(\gamma)$"
        y = r"$| \hat{\rho}^\pi(\gamma) - \rho^\pi(\gamma) |$"
        ylabel = x if not errors else y
        if scale_pv:
            x = ylabel[1:-1]
            y = r"\div (1 - \gamma)"
            ylabel = f"${x} {y}$"
    plt.ylabel(ylabel)

    if legend: plt.legend()
    plt.suptitle(suptitle)
    title = f"{projected=}, {weighted=}"
    plt.title(title)

    plt.ylim(ylim)
    plt.grid(linestyle=":")

    # -------------------------------- #

    if dir is not None:

        if not tf.io.gfile.isdir(dir):
            tf.io.gfile.makedirs(dir)

        file_name = f"{suptitle}; {title}; {legend=}"
        save_path = os.path.join(dir, file_name)

        plt.savefig(save_path, bbox_inches="tight")

    # -------------------------------- #

    plt.show()


def plot_sdc_errors(
        estimators,
        gammas,
        projected,
        modified, lamda,
        #
        sdcs, sdc_lims, sdc_ref, sdc_lim_ref,
        colors, colors_lim,
        markers, markers_lim,
        suptitle,
        one_minus_gamma,
        scale_x=False, scale_y=False, ylim=None, legend=True,
        dir=None):

    if sdc_lims is not None:
        assert sdc_lim_ref is not None

    plt.figure()

    x = gammas if not one_minus_gamma else 1 - gammas

    # -------------------------------- #

    if sdc_lims is not None:

        Z = sdc_lims, colors_lim, markers_lim
        for z in safe_zip(*Z):
            sdc_lim, color, marker = z

            y = np.mean( (sdc_lim - sdc_lim_ref) ** 2 )
            plt.axhline(y, color=color, marker=marker, linestyle=":")

    # ---------------- #

    Z = estimators, sdcs, colors, markers
    for z in safe_zip(*Z):
        estimator, sdc, color, marker = z

        y = np.mean( (sdc - sdc_ref) ** 2, axis=-1 )

        label = estimator.__name__
        if label == "TabularDice":         label += f", {modified=}"
        if label == "TabularGradientDice": label += f", {lamda=}"

        plt.plot(x, y, label=label, color=color, marker=marker)

    # -------------------------------- #

    if scale_x: plt.xscale("log")
    if scale_y: plt.yscale("log")

    if one_minus_gamma: plt.gca().invert_xaxis()

    xlabel = r"$\gamma$" if not one_minus_gamma else r"$1 - \gamma$"
    plt.xlabel(xlabel)

    ylabel = r"$\mathbb{E}_D | \hat{w}_{\pi / D}(\gamma) - w_{\pi / D}(\gamma) |^2$"
    plt.ylabel(ylabel)

    if legend: plt.legend()
    plt.suptitle(suptitle)
    title = f"{projected=}"
    plt.title(title)

    plt.ylim(ylim)
    plt.grid(linestyle=":")

    # -------------------------------- #

    if dir is not None:

        if not tf.io.gfile.isdir(dir):
            tf.io.gfile.makedirs(dir)

        file_name = f"{suptitle}; {title}; {legend=}"
        save_path = os.path.join(dir, file_name)

        plt.savefig(save_path, bbox_inches="tight")

    # -------------------------------- #

    plt.show()

# ---------------------------------------------------------------- #
