#!/usr/bin/env python

import warnings

import arviz as az
import click
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
import theano.tensor as tt
from matplotlib import pyplot as plt
from pymc3 import model_to_graphviz
from sklearn.preprocessing import LabelEncoder

from analysis.plot import plot_trace, plot_neff, plot_rhat

warnings.filterwarnings("ignore")

models = ["shm", "flat"]


def _load_data(infile):
    dat = pd.read_csv(infile, sep="\t")
    dat = (dat[["Condition", "Gene", "sgRNA", "r1", "r2"]]
           .query("Gene != 'Control'")
           .melt(id_vars=["Gene", "Condition", "sgRNA"],
                 value_vars=["r1", "r2"],
                 var_name="replicate",
                 value_name="counts")
           .sort_values(["Gene", "Condition", "sgRNA", "replicate"])
           )
    dat["sgRNA"] = LabelEncoder().fit_transform(dat["sgRNA"].values)
    dat["counts"] = sp.floor(dat["counts"].values)
    return dat


def shm(read_counts: pd.DataFrame):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = sp.unique(read_counts["Condition"].values)
    genes = sp.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))
    len_sirnas = len(sp.unique(read_counts["sgRNA"].values))
    len_replicates = len(sp.unique(read_counts["replicate"].values))
    len_sirnas_per_gene = int(len_sirnas / len_genes)

    beta_idx = np.repeat(range(len_genes), len_conditions)
    gamma_idx = np.repeat()
    beta_data_idx = np.repeat(beta_idx, int(n / len(beta_idx)))

    gene_conds = [a + b for a, b in zip(genes[beta_idx],
                                        conditions[np.repeat(range(len_genes),
                                                             len(conditions))])]

    l_idx = np.repeat(
      range(len_genes * len_conditions * len_sirnas_per_gene), len_replicates)

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

    return model, genes, gene_conds


def flat(read_counts):
    return 1


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option("--model-type", type=click.Choice(models), default="shm")
def run(infile, outfile, model_type):
    read_counts = _load_data(infile)
    read_counts = read_counts.query("Gene == 'POLR3K' | Gene == 'BCR'")

    if model_type == models[0]:
        model, genes, gene_conds = shm(read_counts)
    else:
        model = flat(read_counts)

    n_sample, n_tune, n_init = 1000, 500, 1000

    with model:
        trace = pm.sample(n_sample, tune=n_tune, init="advi", n_init=n_init,
                          chains=4, random_seed=42, progressbar=True,
                          discard_tuned_samples=False)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)

    graph = model_to_graphviz(model)
    graph.render(filename=outfile + ".dot")

    for format in ["pdf", "svg", "eps"]:
        fig, _ = az.plot_forest(trace, var_names="gamma", credible_interval=0.95)
        fig.savefig(outfile + "_forest_gamma." + format)
        fig, _ = az.plot_forest(trace, var_names="beta", credible_interval=0.95)
        fig.savefig(outfile + "_forest_beta." + format)
        fig, _ = az.plot_forest(trace, var_names="category", credible_interval=0.95)
        fig.savefig(outfile + "_forest_category." + format)

        for i, g in enumerate(genes):
            fig, _ = plot_trace(trace, "gamma", n_tune, i, g)
            fig.savefig(outfile + "_trace_gamma_{}_{}.{}".format(i, g, format))


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
