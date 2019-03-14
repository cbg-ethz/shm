
import numpy as np
import pandas as pd


from matplotlib import pyplot as plt
import seaborn as sns
import arviz as az




sns.set_style(
  "white",
  {
      "xtick.bottom": True,
      "ytick.left": True,
      "axes.spines.top": False,
      "axes.spines.right": False,
  },
)


def _plot_dotline(table, boundary, var, low, mid, high, legend, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=720)

    plt.axvline(x=boundary[0], color="grey", linestyle="--")
    plt.axvline(x=boundary[1], color="grey", linestyle="--")
    plt.axvline(x=boundary[2], color="grey", linestyle="--")

    plt.hlines(
      y=table["param"].values[low],
      xmin=np.min(table[var].values),
      xmax=table[var].values[low],
      linewidth=1,
      color="#023858",
    )
    plt.hlines(
      y=table["param"].values[mid],
      xmin=np.min(table[var].values),
      xmax=table[var].values[mid],
      linewidth=1,
      color="#045a8d",
    )
    plt.hlines(
      y=table["param"].values[high],
      xmin=np.min(table[var].values),
      xmax=table[var].values[high],
      linewidth=1,
      color="#74a9cf",
    )

    plt.plot(
      table[var].values[low],
      table["param"].values[low],
      "o",
      markersize=5,
      color="#023858",
      label="${} < {}$".format(legend, boundary[0]),
    )
    plt.plot(
      table[var].values[mid],
      table["param"].values[mid],
      "o",
      markersize=5,
      color="#045a8d",
      label="${} < {}$".format(legend, boundary[1]),
    )
    plt.plot(
      table[var].values[high],
      table["param"].values[high],
      "o",
      markersize=5,
      color="#74a9cf",
      label="${} >= {}$".format(legend, boundary[1]),
    )

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel("Parameters")
    plt.yticks([])

    return fig, ax


def plot_neff(trace, var_name):
    eff_samples = (az.effective_sample_size(trace)[var_name]).to_dataframe()
    boundary = [0.1, 0.5, 1]
    eff_samples = pd.DataFrame({
        "neff": eff_samples[var_name].values / len(trace),
        "param": [var_name + str(i) for i in range(len(eff_samples))]})
    low = np.where(eff_samples["neff"].values < boundary[0])
    mid = np.where(np.logical_and(
        eff_samples["neff"].values >= boundary[0],
        eff_samples["neff"].values < boundary[1]))
    high = np.where(eff_samples["neff"].values >= boundary[1])

    return _plot_dotline(eff_samples, boundary, "neff",
                         low, mid, high,
                         "n_eff / n", "Effective sample size", "n_eff / n")



def plot_rhat(trace, var_name):
    rhat_samples = (az.rhat(trace)[var_name]).to_dataframe()
    boundary = [1.05, 1.1, 1.5]
    rhat_samples = pd.DataFrame({
          "rhat": rhat_samples[var_name].values,
          "param": [var_name + str(i) for i in range(len(rhat_samples))]})
    low = np.where(rhat_samples["rhat"].values < boundary[0])
    mid = np.where(np.logical_and(
        rhat_samples["rhat"].values >= boundary[0],
        rhat_samples["rhat"].values < boundary[1]))
    high = np.where(rhat_samples["rhat"].values >= boundary[1])

    return _plot_dotline(rhat_samples, boundary, "rhat",
                         low, mid, high,
                         r"\hat{R}", "Effective sample size", r"$\hat{R}$")



def _to_df(trace, var_name, idx):
    n_chains = trace.ntune
    samples = trace.get_values(var_name)[:, idx]
    len_per_sample = int(len(samples) / n_chains)
    chains = np.repeat(trace.chains, len_per_sample) + 1
    frame = pd.DataFrame({
        'Chain': chains,
        'sample': samples,
        'idxx': np.tile(range(len_per_sample), n_chains)
    })

    return frame


def plot_trace(trace, var_name, ntune, idx, title):
    frame = _to_df(trace, var_name, idx)

    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=720)
    sns.lineplot(x="idxx", y="sample", data=frame, hue='Chain',
                 palette=sns.cubehelix_palette(4, start=.5, rot=-.75))

    plt.axvline(ntune, linestyle='--', linewidth=.5, color="grey")
    plt.annotate("Burn-in", (ntune, 0))
    plt.legend(bbox_to_anchor=(.95, 0.5), loc="center left", frameon=False)
    plt.xlabel("")
    plt.title(title, loc="Left")

    return fig, ax


def plot_hist(trace, var_name, ntune, idx, title):
    fr = _to_df(trace, var_name, idx)
    fr = fr[["samples", "chain", "idxx"]].pivot(index="idxx", columns="chain")
    fr = fr.values

    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=720)
    cols = sns.cubehelix_palette(4, start=.5, rot=-.75).as_hex()
    ax.hist(fr[ntune:, 0], 50, color=cols[0], label="1", alpha=.75)
    ax.hist(fr[ntune:, 1], 50, color=cols[1], label="2", alpha=.75)
    ax.hist(fr[ntune:, 2], 50, color=cols[2], label="3", alpha=.75)
    ax.hist(fr[ntune:, 3], 50, color=cols[3], label="4", alpha=.75)

    leg = plt.legend(title="Chain", bbox_to_anchor=(.95, 0.5),
                     loc="center left", frameon=False)
    leg._legend_box.align = "left"
    plt.xlabel("")
    plt.title(title, loc="Left")

    return fig, ax

def plot_parallel(trace):
    fig, ax = plt.subplots(figsize=(8, 3), dpi=720)
    az.plot_parallel(trace, ax=ax)
    ax.xaxis.set_ticklabels([])
    return fig, ax