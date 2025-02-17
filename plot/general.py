# ---------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt

from dice_rl_TU_Vienna.utils.numpy import moving_average_Z

# ---------------------------------------------------------------- #

def plot_histogram(
        data, bins,
        suptitle, xlabel,
        moving_average_radius=None,
        xlim=None, ylim=None,
        xscale=False, yscale=False,
    ):

    data_mean = np.mean(data)
    data_std  = np.std(data)

    plt.suptitle(
        f"{suptitle}" + "\n" + ", ".join([
            f"{bins=}",
            "mean" + r"$\approx$" + str(round(data_mean, 3)),
            "std"  + r"$\approx$" + str(round(data_std,  3)),
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
        color="black", linestyles=":",
        label="mean",
    )
    plt.vlines(
        x=data_mean + data_std,
        ymin=0,
        ymax=counts_max,
        color="black", linestyles="--",
        label=f"mean $+$ std",
    )

    plt.grid(linestyle=":")

    if xscale: plt.xscale("log")
    if yscale: plt.yscale("log")

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel(xlabel)
    plt.ylabel("count")

    plt.legend()

    plt.show()

# ---------------------------------------------------------------- #
