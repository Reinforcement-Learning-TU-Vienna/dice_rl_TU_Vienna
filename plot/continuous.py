# ---------------------------------------------------------------- #

import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf

from collections import defaultdict

from dice_rl_TU_Vienna.latex import latex_labels

from utils.general import safe_zip, shape
from utils.numpy import moving_average

from utils.general import safe_zip, list_ify

# ---------------------------------------------------------------- #

def get_logs(log_dir, file_name=None):

    if file_name is None:
        file_names = [f for f in os.listdir(log_dir) if f != ".DS_Store"]
        file_name = file_names[-1]

    event_file_path = os.path.join(log_dir, file_name) # type: ignore
    print(f"Getting log: {event_file_path}")

    summary_iterator = tf.compat.v1.train.summary_iterator(event_file_path)

    logs = {
        "event_file_path": event_file_path,
        "data": defaultdict(lambda: {"steps": [], "values": []})
    }

    for event in summary_iterator:
        if event.HasField('summary'):
            for value in event.summary.value:
                tensor = tf.io.parse_tensor(
                    value.tensor.SerializeToString(), out_type=tf.float32)
                logs["data"][value.tag]["steps"].append(event.step)
                logs["data"][value.tag]["values"].append(float(tensor))

    return logs

# ---------------------------------------------------------------- #

def plot_log(logs, infos, suptitle=None, save_dir=None, file_name=None):
    if type(logs) is dict:
        logs_array = [logs]
        infos_array = [infos]
    elif type(logs) is list:
        logs_array = logs
        infos_array = infos
    else:
        raise NotImplementedError

    n_cols, n_rows = shape(infos_array, depth=2) # type: ignore

    _, ax_mat = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        tight_layout=True,
        figsize=(6 * len(logs_array), 8))

    if suptitle is not None: plt.suptitle(suptitle)

    display_description = True
    for ax_vec, infos, logs in safe_zip(ax_mat.T, infos_array, logs_array):
        colors = ["blue", "orange", "green", "red"]

        for ax, info in safe_zip(ax_vec, infos):
            for i, tag in enumerate(info["tags"]):

                x = logs["data"][tag]["steps"]
                y = logs["data"][tag]["values"]

                xmin = min(x)
                xmax = max(x)

                k = tag
                if tag == "pv_s": k = "pv_approx_s"
                if tag == "pv_w": k = "pv_approx_w"
                label = latex_labels[k]("")
                color = colors[i]
                alpha = 0.1 if "n_moving_averages" in info.keys() else 1

                ax.plot(x, y, label=label, color=color, alpha=alpha)

                if "plot_type" in info.keys():
                    if info["plot_type"] in ["semilogx", "loglog"]: ax.set_xscale("log")
                    if info["plot_type"] in ["semilogy", "loglog"]: ax.set_yscale("log")


                if "n_moving_averages" in info.keys():
                    n_moving_average = info["n_moving_averages"]
                    if n_moving_average is not None:
                        x_ma = x[n_moving_average-1:]
                        y_ma = moving_average(y, n_moving_average)
                        if len(x_ma) == len(y_ma): ax.plot(x_ma, y_ma, color=colors[i])

            if "baselines" in info.keys():
                for baseline in info["baselines"]:
                    ax.hlines(
                        y=baseline["value"], xmin=xmin, xmax=xmax,
                        label=baseline["label"],
                        colors="black", linestyles=baseline["linestyle"])

            if "title" in info.keys():
                ax.set_title(info["title"])

            if "ylabel" in info.keys():
                if display_description:
                    ax.set_ylabel(info["ylabel"])

            if "xlim" in info.keys():
                ax.set_xlim(info["xlim"])

            if "ylim" in info.keys():
                ax.set_ylim(info["ylim"])

            if display_description:
                legend = ax.legend()
                for lh in legend.legend_handles:
                    lh.set_alpha(1)

            ax.grid(linestyle=":")

        display_description = False

        plt.xlabel("step")

    if save_dir is not None:

        if not tf.io.gfile.isdir(save_dir):
            tf.io.gfile.makedirs(save_dir)

        assert file_name is not None
        save_path = os.path.join(save_dir, f"{file_name}.png")

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

# ---------------------------------------------------------------- #

hparam_str_dict = {
    "gam": "gamma",
    "nstep": "number-of-steps",
    "batchs": "batch-size",
    "seed": "seed",
    "hdp": "hidden-dimensions-primal",
    "hdd": "hidden-dimensions-dual",
    "lrp": "learning-rate-primal",
    "lrd": "learning-rate-dual",
    "regp": "mlp-regularizer-primal",
    "regd": "mlp-regularizer-dual",
    "fexp": "f-exponent",
    "nlr": "learning-rate-norm",
    "nreg": "norm-regularizer",
}

def hparam_str_evaluation_to_hparams(hparam_str_evaluation):
    slices = hparam_str_evaluation.split("_")
    hparams = {}
    for slice in slices:

        k = None
        v = None
        for name in hparam_str_dict.keys():
            if name in slice:
                k = name
                v = slice.removeprefix(k)
                break
        assert k is not None and v is not None, f"{slice} from {slices} not found"

        hparams[k] = v

    return hparams

def get_plot_logs(
        get_suptitle, get_pv_baselines,
        #
        outputs_dir,
        hparam_str_policy, hparam_str_dataset,
        estimator_name, hparam_str_evaluation,
        #
        error_tags=None, plot_types=None,
        #
        title=None,
        xlim=None,
        ylim_1=None, ylim_2=None, ylim_3=None,
        n_ma_1=None, n_ma_2=None, n_ma_3=None,
        #
        save_dir=None, file_name=None,
        hparams_title=None,
    ):

    # -------------------------------- #

    if error_tags is None:
        # all = ["pv_error", "sdc_L1_error", "sdc_L2_error", "bellman_L1_error", "bellman_L2_error", "norm_error"]
        error_tags = []
    if plot_types is None:
        plot_types = ["plot", "semilogy", "plot"]

    l = list_ify(
        hparam_str_evaluation, file_name,
        title,
        xlim,
        ylim_1, ylim_2, ylim_3,
        n_ma_1, n_ma_2, n_ma_3,
    )

    d = {
        "hparam_str_evaluation": l[0], "file_name": l[1],
        "title": l[2],
        "xlim": l[3],
        "ylim_1": l[4], "ylim_2": l[5], "ylim_3": l[6],
        "n_ma_1": l[7], "n_ma_2": l[8], "n_ma_3": l[9],
    }

    # -------------------------------- #

    logs = []

    Z = d["hparam_str_evaluation"], d["file_name"]
    for z in safe_zip(*Z):
        hparam_str_evaluation, file_name = z

        log_dir = os.path.join(
            outputs_dir,
            hparam_str_policy, hparam_str_dataset,
            estimator_name, hparam_str_evaluation, )

        log = get_logs(
            log_dir=log_dir,
            file_name=file_name,
        )
        logs.append(log)

    # -------------------------------- #

    gammas = []
    lambdas = []

    for hparam_str_evaluation in d["hparam_str_evaluation"]:

        hparams_evaluation = hparam_str_evaluation_to_hparams(hparam_str_evaluation)

        assert "gam" in hparams_evaluation.keys()
        g = float( hparams_evaluation["gam"] )
        gammas.append(g)

        l = float( hparams_evaluation["nreg"] ) if "nreg" in hparams_evaluation.keys() else None
        lambdas.append(l)

    # -------------------------------- #

    infos = []

    Z = d["title"], d["xlim"], d["ylim_1"], d["ylim_2"], d["ylim_3"], d["n_ma_1"], d["n_ma_2"], d["n_ma_3"], gammas, lambdas
    for z in safe_zip(*Z):
        title, xlim_, ylim_1, ylim_2, ylim_3, n_ma_1, n_ma_2, n_ma_3, g, l = z

        ylabel =[ "policy value", "errors", "loss", ]

        i_1 = { "tags": ["pv_s", "pv_w"], "title": title[0], "ylabel": ylabel[0], "xlim": xlim_, "ylim": ylim_1, "plot_type": plot_types[0], "n_moving_averages": n_ma_1, }
        i_2 = { "tags": error_tags,       "title": title[1], "ylabel": ylabel[1], "xlim": xlim_, "ylim": ylim_2, "plot_type": plot_types[1], "n_moving_averages": n_ma_2, }
        i_3 = { "tags": ["loss"],         "title": title[2], "ylabel": ylabel[2], "xlim": xlim_, "ylim": ylim_3, "plot_type": plot_types[2], "n_moving_averages": n_ma_3, }

        i_1["baselines"] = get_pv_baselines(gamma=g)
        i = [i_1] + [i_2] * ( len(error_tags) > 0 ) + [i_3]

        infos.append(i)

    # -------------------------------- #

    hparams = hparam_str_evaluation_to_hparams(hparam_str_evaluation)

    parts_d  = {}

    parts_d[ hparam_str_dict["gam"] ] = hparams["gam"]

    parts_d[ hparam_str_dict["batchs"] ] = hparams["batchs"]

    hd = [ hparams["hdp"], hparams["hdd"], ]
    assert len( set(hd) ) == 1
    parts_d[ "hidden-dimensions"] = random.choice(hd)

    lr = [ hparams["lrp"], hparams["lrd"], ]
    if "nlr" in hparams.keys(): lr += [ hparams["nlr"] ]
    assert len( set(lr) ) == 1
    parts_d["learning-rate"] = random.choice(lr)

    reg = [ hparams["regp"], hparams["regd"], ]
    assert len( set(reg) ) == 1
    parts_d["mlp-regularizer"] = random.choice(reg)

    parts_l = [ estimator_name, ]
    for k, v in parts_d.items():
        if hparams_title is not None:
            if k not in hparams_title:
                continue
        parts_l.append( f"{k}={v}" )

    suptitle = get_suptitle(gammas)
    subtitle = ", ".join(parts_l)
    title = suptitle + "\n" + subtitle

    file_name = title \
        .replace(" - ", "_") \
        .replace("\n",  "_") \
        .replace(", ",  "_")

    # -------------------------------- #

    plot_log(logs, infos, title, save_dir, file_name)

# ---------------------------------------------------------------- #
