#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore")

import click
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
import theano.tensor as tt

from sklearn import preprocessing
from pymc3 import model_to_graphviz

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

models = ["shm", "flat"]


def _load_data(infile):
    dat = pd.read_csv(infile, sep="\t")
    dat = (
        dat[["Condition", "Gene", "sgRNA", "r1", "r2"]]
            .query("Gene != 'Control'")
            .melt(
          id_vars=["Gene", "sgRNA"],
          value_vars=["r1", "r2"],
          var_name="replicate",
          value_name="counts",
        )
            .sort_values(["Gene", "Condition", "sgRNA", "replicate"])
    )
    dat["sgRNA"] = preprocessing.LabelEncoder.fit_transform(dat["sgRNA"].values)
    dat["counts"] = sp.floor(dat["counts"].values)
    return dat


def _plot_dotline(table, boundary, var, low, mid, high, legend, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=720)

    plt.axvline(x=boundary[0], color="grey", linestyle="--")
    plt.axvline(x=boundary[1], color="grey", linestyle="--")
    plt.axvline(x=boundary[2], color="grey", linestyle="--")

    plt.hlines(
      y=table["param"].values[low],
      xmin=np.min(table["neff"].values),
      xmax=table[var].values[low],
      linewidth=1,
      color="#023858",
    )
    plt.hlines(
      y=table["param"].values[mid],
      xmin=np.min(table["neff"].values),
      xmax=table[var].values[mid],
      linewidth=1,
      color="#045a8d",
    )
    plt.hlines(
      y=table["param"].values[high],
      xmin=np.min(table["neff"].values),
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
    eff_samples = az.effective_sample_size(trace).to_dataframe()
    boundary = [0.1, 0.5, 1]
    eff_samples = pd.DataFrame({
        "neff": eff_samples[[var_name]].values[:, 0] / len(trace),
        "param": [var_name + str(i) for i in range(eff_samples.shape[0])],
    }
    )
    low = np.where(eff_samples["neff"].values < boundary[0])
    mid = np.where(
      np.logical_and(
        eff_samples["neff"].values >= boundary[0],
        eff_samples["neff"].values < boundary[1],
      )
    )
    high = np.where(eff_samples["neff"].values >= boundary[1])

    return _plot_dotline(
      eff_samples,
      boundary,
      "neff",
      low,
      mid,
      high,
      "n_eff / n",
      "Effective sample size",
      "n_eff / n",
    )


def plot_rhat(trace, var_name):
    rhat_samples = az.rhat(trace).to_dataframe()
    boundary = [1.05, 1.1, 1.5]
    rhat_samples = pd.DataFrame(
      {
          "rhat": rhat_samples[[var_name]].values[:, 0],
          "param": [var_name + str(i) for i in range(rhat_samples.shape[0])],
      }
    )
    low = np.where(rhat_samples["rhat"].values < boundary[0])
    mid = np.where(
      np.logical_and(
        rhat_samples["rhat"].values >= boundary[0],
        rhat_samples["rhat"].values < boundary[1],
      )
    )
    high = np.where(rhat_samples["rhat"].values >= boundary[1])

    return _plot_dotline(
      rhat_samples,
      boundary,
      "rhat",
      low,
      mid,
      high,
      r"\hat{R}",
      "Effective sample size",
      r"$\hat{R}$",
    )


def shm(read_counts: pd.DataFrame):
    n, _ = read_counts.shape
    le = preprocessing.LabelEncoder()

    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))
    len_sirnas = len(sp.unique(read_counts["sgRNA"].values))
    len_replicates = len(sp.unique(read_counts["replicate"].values))
    len_sirnas_per_gene = int(len_sirnas / len_genes)

    beta_idx = np.repeat(range(len_genes), len_conditions)
    beta_data_idx = np.repeat(beta_idx, n / len(beta_idx))

    l_idx = np.repeat(
      range(len_genes * len_conditions * len_sirnas_per_gene), len_replicates
    )
    l_idx

    with pm.Model() as model:
        p = pm.Dirichlet("p", a=np.array([1.0, 1.0]), shape=2)
        pm.Potential("p_pot", tt.switch(tt.min(p) < 0.05, -np.inf, 0))
        category = pm.Categorical("category", p=p, shape=len_genes)

        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        mean_g = pm.Normal("mu_g", mu=np.array([0, 0]), sd=0.5, shape=2)
        pm.Potential("m_opot", tt.switch(mean_g[1] - mean_g[0] < 0, -np.inf, 0))
        gamma = pm.Normal("gamma", mean_g[category], tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))
        l = pm.Lognormal("l", 0, 0.25, shape=len_sirnas)

        pm.Poisson(
          "x",
          mu=np.exp(beta[beta_data_idx]) * l[l_idx],
          observed=sp.squeeze(read_counts["counts"].values),
        )

    return model


def flat(read_counts):
    return 1


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option("--model-type", type=click.Choice(models), default="shm")
def run(infile, outfile, model_type):
    read_counts = _load_data(infile)
    read_counts = read_counts.query("Gene == 'POLR3K'")

    if model_type == models[0]:
        model = shm(read_counts)
    else:
        model = flat(read_counts)

    with model:
        trace = pm.sample(
          10000,
          tune=5000,
          init="advi",
          n_init=10000,
          chains=4,
          random_seed=42,
          progressbar=False,
          discard_tuned_samples=False,
        )

    pm.save_trace(trace, outfile + "_trace")

    graph = model_to_graphviz(model)
    graph.render(filename=outfile + ".dot")

    for format in ["pdf", "svg", "eps"]:
        fig, axes = az.plot_trace(trace, var_names=["gamma"])
        fig.savefig(outfile + "_trace_gamma." + format)

        fig, axes = az.plot_trace(trace, var_names=["category"])
        fig.savefig(outfile + "_trace_category." + format)

        fig, axes = az.plot_trace(trace, var_names=["beta"])
        fig.savefig(outfile + "_trace_beta." + format)

        fig, axes = az.plot_forest(
          trace, credible_interval=0.95,
          var_names=["beta", "gamma", "category"]
        )
        fig.savefig(outfile + "_forest." + format)
        fig, ax = plot_neff(trace, "gamma")
        fig.savefig(outfile + "_neff_gamma." + format)
        fig, ax = plot_neff(trace, "beta")
        fig.savefig(outfile + "_neff_beta." + format)
        fig, ax = plot_neff(trace, "category")
        fig.savefig(outfile + "_neff_category." + format)
        fig, ax = plot_rhat(trace, "gamma")
        fig.savefig(outfile + "_rhat_gamma." + format)
        fig, ax = plot_rhat(trace, "beta")
        fig.savefig(outfile + "_rhat_beta." + format)
        fig, ax = plot_rhat(trace, "category")
        fig.savefig(outfile + "_rhat_category." + format)


if __name__ == "__main__":
    run()
