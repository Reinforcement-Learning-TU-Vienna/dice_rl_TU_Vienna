# ---------------------------------------------------------------- #

import os

import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------------------------------------------------------- #

def plot(
        plot_infos,
        suptitle,
        one_minus_gamma=True, scale_pv=False,
        xlabel=None, ylabel=None,
        xlim=None, ylim=None,
        xscale=True, yscale=False,
        legend=True,
        dir=None,
    ):

    plt.figure(tight_layout=True)
    plt.suptitle(suptitle)

    for plot_info in plot_infos:
        A = "x" in plot_info.keys()
        B = "y" in plot_info.keys()
        if A and B:
            plt.plot(
                plot_info["x"] if not one_minus_gamma else 1 - plot_info["x"],
                plot_info["y"] if not scale_pv else plot_info["y"] / (1 - plot_info["x"]),
                label=plot_info["label"],
                color=plot_info["color"],
                marker=plot_info["marker"],
            )

        else:
            if A:
                plt.axvline(
                    plot_info["x"] if not one_minus_gamma else 1 - plot_info["x"],
                    label=plot_info["label"],
                    color=plot_info["color"],
                    marker=plot_info["marker"],
                    linestyle=":",
                )
            if B:
                plt.axhline(
                    plot_info["y"],
                    label=plot_info["label"],
                    color=plot_info["color"],
                    marker=plot_info["marker"],
                    linestyle=":",
                )

    if xscale: plt.xscale("log")
    if yscale: plt.yscale("log")

    if one_minus_gamma: plt.gca().invert_xaxis()

    if xlabel is None:
        x = r"\gamma"
        if one_minus_gamma: x = f"1 - {x}"
        xlabel = f"${x}$"

    if ylabel is None:
        ylabel = ""

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(xlim)
    plt.ylim(ylim)

    if legend:
        if   isinstance(legend, bool): loc = "best"
        elif isinstance(legend, dict): loc = legend.get("loc", "best")
        else: raise TypeError
        plt.legend(loc=loc)

    plt.grid(linestyle=":")

    if dir is not None:

        if not tf.io.gfile.isdir(dir):
            tf.io.gfile.makedirs(dir)

        file_name = suptitle.replace("\n", "; ")
        save_path = os.path.join(dir, file_name)

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

# ---------------------------------------------------------------- #
