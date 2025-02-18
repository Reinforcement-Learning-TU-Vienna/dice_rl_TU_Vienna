# ---------------------------------------------------------------- #

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from collections import defaultdict

from dice_rl_TU_Vienna.latex import latex_labels
from dice_rl_TU_Vienna.utils.general import safe_zip, shape, list_ify
from dice_rl_TU_Vienna.utils.numpy import moving_average_N
from dice_rl_TU_Vienna.utils.json import json_get_id
from dice_rl_TU_Vienna.utils.os import os_path_join

# ---------------------------------------------------------------- #

def get_log(log_dir, file_name=None, verbosity=0):

    if file_name is None:
        file_names = [f for f in os.listdir(log_dir) if f != ".DS_Store"]
        file_name = file_names[-1]

    event_file_path = os.path.join(log_dir, file_name) # type: ignore
    if verbosity > 0: print(f"Getting log {event_file_path}")

    summary_iterator = tf.compat.v1.train.summary_iterator(event_file_path)

    log = {
        "event_file_path": event_file_path,
        "data": defaultdict(lambda: {"steps": [], "values": []})
    }

    for event in summary_iterator:
        if event.HasField("summary"):
            for value in event.summary.value:
                tensor = tf.io.parse_tensor(
                    value.tensor.SerializeToString(), out_type=tf.float32)

                log["data"][value.tag]["steps"].append(event.step)
                log["data"][value.tag]["values"].append(float(tensor))

    return log

# ---------------------------------------------------------------- #

def plot(infos, suptitle=None, dir_save=None, file_name=None):

    n_cols, n_rows = shape(infos, depth=2) # type: ignore

    _, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        tight_layout=True,
        figsize=(5*n_cols, 3*n_rows),
    )

    for i, (info_column, ax_column) in enumerate( safe_zip(infos, axs.T) ):
        for j, (info_row, ax_row) in enumerate( safe_zip(info_column, ax_column) ):

            for info_hline in info_row.get("hlines", []):
                ax_row.hlines(
                    y=info_hline["y"],
                    xmin=info_hline["xmin"],
                    xmax=info_hline["xmax"],
                    label=info_hline["label"],
                    colors=info_hline["color"],
                    linestyles=info_hline["linestyle"],
                )

            for info_plot in info_row.get("plots", []):
                ax_row.plot(
                    info_plot["x"],
                    info_plot["y"],
                    label=info_plot["label"],
                    color=info_plot["color"],
                    alpha=info_plot["alpha"],
                )

            if ( plot_type := info_row.get("plot_type", None) ) is not None:
                if plot_type in ["semilogx", "loglog"]: ax_row.set_xscale("log")
                if plot_type in ["semilogy", "loglog"]: ax_row.set_yscale("log")

            if ( title := info_row.get("title", None) ) is not None:
                ax_row.set_title(title)

            if ( xlabel := info_row.get("xlabel", None) ) is not None:
                ax_row.set_xlabel(xlabel)

            if ( ylabel := info_row.get("ylabel", None) ) is not None:
                ax_row.set_ylabel(ylabel)

            if ( xlim := info_row.get("xlim", None) ) is not None:
                ax_row.set_xlim(xlim)

            if ( ylim := info_row.get("ylim", None) ) is not None:
                ax_row.set_ylim(ylim)

            ax_row.grid(linestyle=":")
            if i == 0: ax_row.legend()

    if suptitle is not None: plt.suptitle(suptitle)

    if dir_save is not None:
        assert file_name is not None

        if not tf.io.gfile.isdir(dir_save):
            tf.io.gfile.makedirs(dir_save)

        save_path = os.path.join(dir_save, f"{file_name}.png")
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

# ---------------------------------------------------------------- #

def get_logs_from_hyperparameters(
        dir_base,
        hyperparameters_evaluation,
        hyperparameters_dict=None,
        verbosity=0):

    if hyperparameters_dict is None: hyperparameters_dict = {}

    dir_log = dir_base

    for name, hyperparameters in hyperparameters_dict.items():
        file_dir = os.path.join(dir_log, f"{name}.json")
        dictionary = hyperparameters
        id = json_get_id(file_dir, dictionary)
        assert id is not None
        dir_log = os.path.join(dir_log, id)

    if not isinstance(hyperparameters_evaluation, list):
        hyperparameters_evaluation = [ hyperparameters_evaluation ]

    logs = []
    file_dir = os.path.join(dir_log, "evaluation.json")
    for dictionary in hyperparameters_evaluation:
        id_evaluation = json_get_id(file_dir, dictionary)
        assert id_evaluation is not None
        log_dir = os.path.join(dir_log, id_evaluation)

        log = get_log(
            log_dir=log_dir,
            verbosity=verbosity,
        )
        logs.append(log)

    return logs

colors = { False: "blue", True: "orange", }

def append_pv_(weighted, info_row, i_log, log, n_samples_moving_average):
    tag = "pv_s" if not weighted else "pv_w"
    tag_label = "pv_approx_s" if not weighted else "pv_approx_w"

    ns_ma = n_samples_moving_average[i_log].get(tag, None)
    use_ma = ns_ma is not None

    x = log["data"][tag]["steps"]
    y = log["data"][tag]["values"]
    label = latex_labels[tag_label]("")

    info_plot = {
        "x": x, "y": y,
        "label": None if use_ma else label,
        "color": "blue",
        "alpha": 0.1 if use_ma else 1,
        "plot_type": None,
    }
    info_row["plots"].append(info_plot)

    if use_ma:
        x_ma = x[ns_ma-1:]
        y_ma = moving_average_N(y, ns_ma)
        info_plot = {
            "x": x_ma, "y": y_ma,
            "label": label,
            "color": colors[weighted],
            "alpha": 1,
            "plot_type": None,
        }
        info_row["plots"].append(info_plot)

    return x

def append_analytical(info_row, i_log, hlines, x_s, x_w):
    x = [x_s, x_w][ np.random.randint(2) ]
    for hline in hlines[i_log].get("pv", []):
        info_hline = {
            "y": hline["y"],
            "xmin": np.min(x),
            "xmax": np.max(x),
            "label": hline["label"],
            "color": "black",
            "linestyle": hline["linestyle"],
        }
        info_row["hlines"].append(info_hline)

def append_pv(
        info_column,
        i_log, log,
        titles, ylims, n_samples_moving_average, hlines):

    info_row = {}

    info_row["plots"] = []
    info_row["hlines"] = []

    args = [ info_row, i_log, log, n_samples_moving_average, ]
    x_s = append_pv_(True,  *args)
    x_w = append_pv_(False, *args)

    append_analytical(info_row, i_log, hlines, x_s, x_w)

    info_row["title"] = titles[i_log]["pv"]
    info_row["xlabel"] = "step" if i_log == 0 else None
    info_row["ylabel"] = "policy value"
    info_row["ylim"] = None if ylims is None else ylims[i_log].get("pv", None)

    info_column.append(info_row)

def append_loss(
        info_column,
        i_log, log,
        titles, ylims, n_samples_moving_average, hlines):

    info_row = {}

    info_row["plots"] = []

    ns_ma = None if n_samples_moving_average is None else n_samples_moving_average[i_log].get("loss", None)
    use_ma = ns_ma is not None

    x = log["data"]["loss"]["steps"]
    y = log["data"]["loss"]["values"]
    label = latex_labels["loss"]("")

    info_plot = {
        "x": x, "y": y,
        "label": None if use_ma else label,
        "color": "blue",
        "alpha": 0.1 if use_ma else 1,
    }
    info_row["plots"].append(info_plot)

    if use_ma:
        x_ma = x[ns_ma-1:]
        y_ma = moving_average_N(y, ns_ma)
        info_plot = {
            "x": x_ma, "y": y_ma,
            "label": label,
            "color": "blue",
            "alpha": 1,
        }
        info_row["plots"].append(info_plot)

    info_row["ylabel"] = "loss"
    info_row["ylim"] = None if ylims is None else ylims[i_log].get("loss", None)

    info_column.append(info_row)

def get_logs_and_plot(
        dir_base,
        #
        hyperparameters_evaluation,
        hyperparameters_dict=None,
        #
        suptitle=None,
        titles=None,
        ylims=None,
        n_samples_moving_average=None,
        hlines=None,
        #
        append_extras=None,
        #
        dir_save=None, file_name=None,
        verbosity=0,
    ):

    if append_extras is None: append_extras = []

    logs = get_logs_from_hyperparameters(
        dir_base,
        hyperparameters_evaluation,
        hyperparameters_dict,
        verbosity,
    )

    infos = []
    for i_log, log in enumerate(logs):
        info_column = []

        args = [
            info_column,
            i_log, log,
            titles, ylims, n_samples_moving_average, hlines,
        ]

        append_pv(*args)
        append_loss(*args)

        for append_extra in append_extras:
            append_extra(*args)

        infos.append(info_column)

    plot(infos, suptitle, dir_save, file_name)

# ---------------------------------------------------------------- #
