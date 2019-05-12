import numpy as np
import pandas as pd
from arviz import convert_to_dataset
from arviz.plots.plot_utils import xarray_to_ndarray, get_coords

from matplotlib import pyplot as plt
import seaborn as sns
import arviz as az

from shm.globals import READOUT

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
                  legend, title, xlabel, xlim):
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
             label="${} >= {}$".format(legend, boundary[1]))

    plt.title(title, loc="Left")
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel("Parameters")
    plt.tick_params(axis=None)
    plt.xlim(left=xlim)
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
        excluded_vars = [i[1:] for i in var_names if i.startswith("~") and i not in all_vars]
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
    eff_samples = (az.effective_sample_size(trace)[var_name]).to_dataframe()
    boundary = [0.1, 0.5, 1]
    eff_samples = pd.DataFrame({
        "neff": eff_samples[var_name].values / (len(trace) * 4),
        "param": [var_name + str(i) for i in range(len(eff_samples))]})
    if variable is not None:
        eff_samples["param"] = variable
    low = np.where(eff_samples["neff"].values < boundary[0])
    mid = np.where(np.logical_and(
      eff_samples["neff"].values >= boundary[0],
      eff_samples["neff"].values < boundary[1]))
    high = np.where(eff_samples["neff"].values >= boundary[1])

    return _plot_dotline(eff_samples, boundary, "neff",
                         low, mid, high,
                         "n_eff / n", "Effective sample size", "n_eff / n", 0)


def plot_rhat(trace, var_name, variable=None):
    rhat_samples = (az.rhat(trace)[var_name]).to_dataframe()
    boundary = [1.05, 1.1, 1.5]
    rhat_samples = pd.DataFrame({
        "rhat": rhat_samples[var_name].values,
        "param": [var_name + str(i) for i in range(len(rhat_samples))]})
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
                         xlim=1)


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


def plot_hist(trace, var_name, idx, title=""):
    fr = _to_df(trace, var_name, idx)
    fr = fr[["sample", "Chain", "idxx"]].pivot(index="idxx", columns="Chain")
    fr = fr.values

    fig, ax = plt.subplots()
    cols = sns.cubehelix_palette(4, start=.5, rot=-.75).as_hex()
    ax.hist(fr[:, 0], 50, color=cols[0], label="1", alpha=.75)
    ax.hist(fr[:, 1], 50, color=cols[1], label="2", alpha=.75)
    ax.hist(fr[:, 2], 50, color=cols[2], label="3", alpha=.75)
    ax.hist(fr[:, 3], 50, color=cols[3], label="4", alpha=.75)

    leg = plt.legend(title="Chain", bbox_to_anchor=(.95, 0.5),
                     loc="center left", frameon=False)
    leg._legend_box.align = "left"
    plt.xlabel("")
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


def plot_posterior_labels(trace, genes, cols=["#E84646", "#316675"]):
    P1 = np.mean(trace['z'], 0)
    P0 = 1 - P1
    len_z = len(P1)
    prob_table = pd.DataFrame({
        "Probability": np.concatenate((P0, P1)),
        "o": np.append(np.repeat("No-hit", len_z),
                            np.repeat("Hit", len_z)),
        "Gene": np.tile(genes, 2)})
    ax = sns.barplot(x="Gene", y="Probability", hue="o",
                     data=prob_table, palette=cols,
                     linewidth=2.5, edgecolor=".2")
    ax.set_ylim(0, 1)
    sns.despine()
    plt.title('Posterior class labels', loc='left', fontsize=16)
    plt.legend(loc='center right', fancybox=False, framealpha=0, shadow=False,
               borderpad=1, bbox_to_anchor=(1.5, 0.5), ncol=1)
    return ax
