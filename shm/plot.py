import numpy
import numpy as np
import pandas as pd
from arviz import convert_to_dataset
from arviz.plots.plot_utils import xarray_to_ndarray, get_coords

from matplotlib import pyplot as plt
import seaborn as sns
import arviz as az

from shm.diagnostics import rhat, n_eff, cut_rhat, cut_neff
from shm.globals import READOUT
from shm.util import compute_posterior_probabilities

sns.set_style(
  "white",
  {
      "xtick.bottom": True,
      "ytick.left": True,
      "axes.spines.top": False,
      "axes.spines.right": False,
  },
)


def _plot_dotline(table, boundary, var, low, mid, high,
                  legend, title, xlabel, xlim, xticks):
    fig, ax = plt.subplots()

    plt.axvline(x=boundary[0], color="grey", linestyle="--", linewidth=.5)
    plt.axvline(x=boundary[1], color="grey", linestyle="--", linewidth=.5)
    plt.axvline(x=boundary[2], color="grey", linestyle="--", linewidth=.5)

    plt.hlines(y=table["param"].values[low],
               xmin=xlim, xmax=table[var].values[low],
               linewidth=.5, color="#023858")
    plt.hlines(y=table["param"].values[mid],
               xmin=xlim, xmax=table[var].values[mid],
               linewidth=.5, color="#045a8d")
    plt.hlines(y=table["param"].values[high],
               xmin=xlim, xmax=table[var].values[high],
               linewidth=.5, color="#74a9cf")

    plt.plot(table[var].values[low], table["param"].values[low],
             "o", markersize=3, color="#023858",
             label="${} < {}$".format(legend, boundary[0]))
    plt.plot(table[var].values[mid], table["param"].values[mid],
             "o", markersize=3, color="#045a8d",
             label="${} < {}$".format(legend, boundary[1]))
    plt.plot(table[var].values[high], table["param"].values[high],
             "o", markersize=3, color="#74a9cf",
             label="${} \geq {}$".format(legend, boundary[1]))
    plt.title(title, loc="Left")
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel("Parameters")
    plt.tick_params(axis=None)
    plt.yticks([])
    plt.xlim(left=xlim)
    plt.xticks(xticks, xticks)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    plt.tight_layout()
    return fig, ax


def _to_df(trace, var_name, idx):
    n_chains = trace.nchains
    samples = trace.get_values(var_name)[:, idx]
    len_per_sample = int(len(samples) / n_chains)
    chains = np.repeat(trace.chains, len_per_sample) + 1
    frame = pd.DataFrame({
        'Chain': chains,
        'sample': samples,
        'idxx': np.tile(range(len_per_sample), n_chains)
    })

    return frame


def _var_names(var_names, data):
    """Handle var_names input across arviz.
    Parameters
    ----------
    var_names: str, list, or None
    data : xarray.Dataset
        Posterior data in an xarray
    Returns
    -------
    var_name: list or None
    """
    if var_names is not None:

        if isinstance(var_names, str):
            var_names = [var_names]
        if isinstance(data, (list, tuple)):
            all_vars = []
            for dataset in data:
                dataset_vars = list(dataset.data_vars)
                for var in dataset_vars:
                    if var not in all_vars:
                        all_vars.append(var)
        else:
            all_vars = list(data.data_vars)
        excluded_vars = [i[1:] for i in var_names if
                         i.startswith("~") and i not in all_vars]
        if excluded_vars:
            var_names = [i for i in all_vars if i not in excluded_vars]
    return var_names


def _extract(trace):
    divergent_data = convert_to_dataset(trace, group="sample_stats")
    _, diverging_mask = xarray_to_ndarray(
      divergent_data, var_names=("diverging",), combined=True)
    diverging_mask = np.squeeze(diverging_mask)

    posterior_data = convert_to_dataset(trace, group="posterior")
    var_names = _var_names(["beta", "gamma", "tau_b", "tau_g"], posterior_data)
    var_names, _posterior = xarray_to_ndarray(
      get_coords(posterior_data, {}), var_names=var_names, combined=True)
    return diverging_mask, var_names, _posterior


def plot_neff(trace, var_name, variable=None):
    eff_samples = n_eff(trace, var_name)
    boundary = [0.1, 0.5, 1]
    if variable is not None:
        eff_samples["param"] = variable
    low = np.where(eff_samples["neff"].values < boundary[0])
    mid = np.where(np.logical_and(
      eff_samples["neff"].values >= boundary[0],
      eff_samples["neff"].values < boundary[1]))
    high = np.where(eff_samples["neff"].values >= boundary[1])

    return _plot_dotline(eff_samples, boundary, "neff",
                         low, mid, high,
                         "n_eff / n", "Effective sample size", "n_eff / n",
                         -.1, boundary)


def plot_rhat(trace, var_name, variable=None):
    boundary = [1.05, 1.1, 1.5]
    rhat_samples = rhat(trace, var_name)
    if variable is not None:
        rhat_samples["param"] = variable
    low = np.where(rhat_samples["rhat"].values < boundary[0])
    mid = np.where(np.logical_and(
      rhat_samples["rhat"].values >= boundary[0],
      rhat_samples["rhat"].values < boundary[1]))
    high = np.where(rhat_samples["rhat"].values >= boundary[1])

    return _plot_dotline(rhat_samples, boundary, "rhat",
                         low, mid, high, r"\hat{R}",
                         "Potential scale reduction factor", r"$\hat{R}$",
                         0.95, [1] + boundary)


def _plot_bar(data, title):
    fig, ax = plt.subplots()
    sns.barplot(x="cut", y="bins", data=data, color="#045a8d", ax=ax)
    ax.spines['bottom'].set_visible(False)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title(title, loc="Left")
    plt.tight_layout()
    return fig, ax


def plot_neff_bar(trace, var_name):
    data = cut_neff(trace, var_name)
    return _plot_bar(data, "Effective sample size")


def plot_rhat_bar(trace, var_name):
    data = cut_rhat(trace, var_name)
    return _plot_bar(data, r"Rank $\hat{R}$")


def plot_parallel(trace):
    diverging_mask, var_names, _posterior = _extract(trace)
    var_names = [var.replace("\n", " ") for var in var_names]

    diverging_mask = diverging_mask
    _posterior = _posterior

    fig, ax = plt.subplots(dpi=720)
    ax.plot(_posterior[:, ~diverging_mask], color="black", alpha=0.025, lw=.25)

    if np.any(diverging_mask):
        ax.plot(_posterior[:, diverging_mask], color="darkred", lw=.25)

    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names)
    plt.xticks(rotation=90)
    ax.plot([], color="black", label="non-divergent")
    if np.any(diverging_mask):
        ax.plot([], color="darkred", label="divergent")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


def plot_trace(trace, var_name, idx, title=""):
    frame = _to_df(trace, var_name, idx)

    fig, ax = plt.subplots(dpi=720)
    sns.lineplot(x="idxx", y="sample", data=frame, hue='Chain',
                 palette=sns.cubehelix_palette(4, start=.5, rot=-.75))
    plt.legend(title='Chain', bbox_to_anchor=(.95, 0.5), loc="center left",
               frameon=False, labels=['1', '2', '3', '4'])
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title, loc="Left")

    return fig, ax


def plot_hist(trace, var_name, idx, title="", bins=60):
    fr = _to_df(trace, var_name, idx)
    fr = fr[["sample", "Chain", "idxx"]].pivot(index="idxx", columns="Chain")
    fr = fr.values
    n, p = fr.shape
    fig, ax = plt.subplots()

    cols = sns.cubehelix_palette(p, start=.5, rot=-.75).as_hex()
    me = np.round(np.mean(fr), 2)
    for i in range(p):
        ax.hist(fr[:, i], color=cols[i], label=str(i + 1), alpha=.75, bins=bins)
    plt.axvline(x=me, color="grey", linestyle="--", linewidth=.5)

    leg = plt.legend(title="Chain", bbox_to_anchor=(.95, 0.5),
                     loc="center left", frameon=False)
    leg._legend_box.align = "left"
    plt.xlabel("")
    plt.xticks([me], [me])
    plt.ylabel("")
    plt.title(title, loc="Left")

    return fig, ax


def plot_steps(data, ppc_trace=None, bins=50, histtype="step"):
    fig, ax = plt.subplots()
    ax.hist(data[READOUT].values, bins=bins, lw=2, density=True,
            edgecolor='black', histtype=histtype, color='grey',
            label='Data')
    if ppc_trace:
        ax.hist(np.mean(ppc_trace['x'], 0), bins=bins, lw=2, density=True,
                edgecolor='#316675', histtype=histtype, color='grey',
                label='Posterior predictive distribution')
    ax.set_xlabel(r"Log fold-change")
    ax.set_ylabel(r"Density")
    ax.xaxis.set_label_coords(.95, -0.115)
    ax.yaxis.set_label_coords(-0.075, .95)
    if ppc_trace:
        ax.legend(frameon=False)
    return fig, ax


def plot_forest(trace, variable, var_name=None):
    fig, ax = az.plot_forest(trace, var_names=variable, credible_interval=0.95)
    ax[0].set_title('')
    ax[0].set_title('95% credible intervals', size=15, loc="left")
    ax[0].spines['left'].set_visible(True)
    if var_name is not None:
        ax[0].set_yticklabels(var_name)
        ax[0].tick_params()
    return fig, ax


def plot_posterior_labels(trace, genes):
    probs = compute_posterior_probabilities(trace)

    bars = np.add(probs[:, 0], probs[:, 1]).tolist()
    pos = numpy.arange(probs.shape[0])
    barWidth = .5

    fig, ax = plt.subplots()
    ax.bar(pos, probs[:, 0], color='#E84646', edgecolor='black',
            width=barWidth, label="Dependency factor")
    ax.bar(pos, probs[:, 1], bottom=probs[:, 0], color='lightgrey',
            edgecolor='black', width=barWidth, label="Neutral")
    ax.bar(pos, probs[:, 2], bottom=bars, color='#316675', edgecolor='black',
            width=barWidth, label="Restriction factor")
    ax.set_facecolor('white')
    plt.tick_params('both', left=True)
    plt.ylim([-0.05, 1.05])
    plt.yticks([0, .25, .5, .75, 1])
    plt.xticks(pos, genes, rotation=90, fontsize=10)
    plt.title('Posterior class labels', loc='left', fontsize=16)
    plt.legend(loc='center right', fancybox=False, framealpha=0,
               shadow=False,
               borderpad=1, bbox_to_anchor=(1.25, 0.5), ncol=1)
    return ax


def plot_confusion_matrix(confusion_matrix, class_names, fontsize=14):
    df_cm = pd.DataFrame(
      confusion_matrix, index=class_names, columns=class_names)

    fig, ax = plt.subplots()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d", cbar=False, ax=ax)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 va='center',
                                 fontsize=fontsize - 2)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 fontsize=fontsize - 2)
    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.title("Confusion matrix", loc='left', fontsize=fontsize + 2)
    return fig, ax
