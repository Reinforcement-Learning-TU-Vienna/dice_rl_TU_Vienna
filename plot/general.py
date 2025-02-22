# ---------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt

from dice_rl_TU_Vienna.utils.numpy import moving_average_Z

from matplotlib.lines import Line2D

# ---------------------------------------------------------------- #

def plot_histogram(
        data, bins,
        suptitle, xlabel,
        moving_average_radius=None,
        xlim=None, ylim=None,
        xscale=False, yscale=False,
    ):

    data_mean   = np.mean(data)
    data_std    = np.std(data)
    data_median = np.median(data)

    plt.figure(figsize=(10, 5))

    plt.suptitle(
        f"{suptitle}" + "\n" + ", ".join([
            f"{bins=}",
            "mean"   + r"$\approx$" + np.format_float_scientific(data_mean,   precision=2),
            "std"    + r"$\approx$" + np.format_float_scientific(data_std,    precision=2),
            "median" + r"$\approx$" + np.format_float_scientific(data_median, precision=2),
        ])
    )

    if xscale:
        bins = np.logspace(
            np.log10(np.min(data)), np.log10(np.max(data)), bins)

    counts, bins, _ = plt.hist(
        data,
        bins=bins,
        alpha=0.25, color="blue",
    )

    counts_max  = np.max(counts)

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

    marker_legend_mean   = Line2D([0], [0], color="black", marker="+", linestyle=":", label="mean")
    marker_legend_median = Line2D([0], [0], color="black", marker="x", linestyle=":", label="median")

    plt.scatter(x=data_mean,   y=np.sqrt(counts_max), color="black", marker="+")
    plt.scatter(x=data_median, y=np.sqrt(counts_max), color="black", marker="x")

    plt.grid(linestyle=":")

    if xscale: plt.xscale("log")
    if yscale: plt.yscale("log")

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel(xlabel)
    plt.ylabel("count")

    plt.legend(handles=[marker_legend_mean, marker_legend_median])

    plt.show()

# ---------------------------------------------------------------- #
