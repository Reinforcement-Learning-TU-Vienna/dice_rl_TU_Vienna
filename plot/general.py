# ---------------------------------------------------------------- #

import os

import numpy as np
import matplotlib.pyplot as plt

from dice_rl_TU_Vienna.utils.numpy import moving_average_Z

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------- #

def plot_histogram(
        data, bins,
        suptitle, xlabel,
        moving_average_radius=None,
        xlim=None, ylim=None,
        xscale=False, yscale=False,
        true_mean=None,
        dir=None,
    ):

    data_mean   = np.mean(data)
    data_std    = np.std(data)
    data_median = np.median(data)
    data_min    = np.min(data)
    data_max    = np.max(data)

    u, c = np.unique(data, return_counts=True)
    n_eq_0 = np.sum(c[u == 0])
    n_le_0 = np.sum(c[u < 0])
    n_ge_0 = np.sum(c[u > 0])

    info = ", ".join([
        "mean"   + r"$\approx$" + np.format_float_scientific(data_mean,   precision=2),
        "std"    + r"$\approx$" + np.format_float_scientific(data_std,    precision=2),
        "median" + r"$\approx$" + np.format_float_scientific(data_median, precision=2),
        "min"    + r"$\approx$" + np.format_float_scientific(data_min,    precision=2),
        "max"    + r"$\approx$" + np.format_float_scientific(data_max,    precision=2),
    ]) + "\n" + ", ".join([
        r"$\# \{ i \mid x_i < 0 \} = " + str(n_le_0) + "$",
        r"$\# \{ i \mid x_i = 0 \} = " + str(n_eq_0) + "$",
        r"$\# \{ i \mid x_i > 0 \} = " + str(n_ge_0) + "$",
    ])

    plt.figure(figsize=(12, 6), tight_layout=True)

    subtitle = f"{bins=}"
    plt.suptitle(suptitle + "\n" + subtitle)

    if xscale:
        data = data[data > 0]
        bins = np.logspace(
            np.log10(np.min(data)), np.log10(np.max(data)), bins)

    counts, bins, _ = plt.hist(
        data,
        bins=bins,
        alpha=0.25, color="blue",
    )

    counts_max = np.max(counts)

    if moving_average_radius:
        radius = moving_average_radius
        x = bins[radius:-1-radius]
        y = moving_average_Z(counts, radius)
        counts_max = np.max(y)
        plt.plot(
            x, y,
            color="blue",
            label=f"moving average, {radius=}"
        )

    plt.vlines(
        x=data_mean,
        ymin=0,
        ymax=counts_max,
        color="black",
        linestyles=":",
        label="mean",
    )
    plt.vlines(
        x=data_median,
        ymin=0,
        ymax=counts_max,
        color="black",
        linestyles=":",
        label="median",
    )
    plt.vlines(
        x=data_min,
        ymin=0,
        ymax=counts_max,
        color="black",
        linestyles=":",
        label="min",
    )
    plt.vlines(
        x=data_max,
        ymin=0,
        ymax=counts_max,
        color="black",
        linestyles=":",
        label="max",
    )
    if true_mean is not None:
        legend_true_mean = plt.vlines(
            x=1,
            ymin=0,
            ymax=counts_max,
            color="black",
            linestyles=":",
            label="true mean",
        )

    marker_legend_mean   = Line2D([0], [0], color="black", marker="+", linestyle=":", label="mean")
    marker_legend_median = Line2D([0], [0], color="black", marker="x", linestyle=":", label="median")
    marker_legend_min    = Line2D([0], [0], color="black", marker="1", linestyle=":", label="min")
    marker_legend_max    = Line2D([0], [0], color="black", marker="2", linestyle=":", label="max")

    y = counts_max # np.sqrt(counts_max) if yscale else counts_max / 2
    plt.scatter(x=data_mean,   y=y, color="black", marker="+")
    plt.scatter(x=data_median, y=y, color="black", marker="x")
    plt.scatter(x=data_min,    y=y, color="black", marker="1")
    plt.scatter(x=data_max,    y=y, color="black", marker="2")

    plt.grid(linestyle=":")

    if xscale: plt.xscale("log")
    if yscale: plt.yscale("log")

    plt.xlim(xlim)
    plt.ylim(ylim)

    if not yscale: plt.gca().yaxis.set_major_locator( MaxNLocator(integer=True) )

    plt.xlabel(xlabel + "\n"*2 + info)
    plt.ylabel("count")

    handles = [
        marker_legend_mean, marker_legend_median,
        marker_legend_min, marker_legend_max,
    ]
    if true_mean is not None:
        handles.append(legend_true_mean) # type: ignore
    plt.legend(handles=handles)

    if dir is not None:
        file_name = suptitle + "; " + subtitle.replace("$\\approx$", "=")
        save_path = os.path.join(dir, f"{file_name}.png")
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

# ---------------------------------------------------------------- #
