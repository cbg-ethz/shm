#!/usr/bin/env python

import warnings

import arviz as az
import click
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
import theano.tensor as tt
from pymc3 import model_to_graphviz
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from plot import plot_trace, plot_neff, plot_rhat, plot_parallel, plot_hist

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
    beta_data_idx = np.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[np.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

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


def flat(read_counts: pd.DataFrame):
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
    beta_data_idx = np.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[np.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    l_idx = np.repeat(
      range(len_genes * len_conditions * len_sirnas_per_gene), len_replicates)

    with pm.Model() as model:
        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        gamma = pm.Normal("gamma", 0, tau_g, shape=len_genes)

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


def _plot_forest(trace, outfile, genes, gene_cond, fm, model):
    fig, _ = az.plot_forest(trace, var_names="gamma", credible_interval=0.95)
    _[0].set_title('')
    _[0].set_title('95% credible intervals', size=15, loc="left")
    _[0].spines['left'].set_visible(True)
    _[0].set_yticklabels(genes)
    _[0].tick_params()
    fig.savefig(outfile + "_forest_gamma." + fm)
    fig, _ = az.plot_forest(trace, var_names="beta", credible_interval=0.95)
    _[0].set_title('')
    _[0].set_title('95% credible intervals', size=15, loc="left")
    _[0].spines['left'].set_visible(True)
    _[0].set_yticklabels(gene_cond)
    _[0].tick_params()
    fig.savefig(outfile + "_forest_beta." + fm)
    if model != "flat":
        fig, _ = az.plot_forest(trace, var_names="category", credible_interval=0.95)
        _[0].set_title('')
        _[0].set_title('95% credible intervals', size=15, loc="left")
        _[0].spines['left'].set_visible(True)
        _[0].tick_params()
        fig.savefig(outfile + "_forest_category." + fm)
    plt.close('all')


def _plot_trace(trace, outfile, n_tune, genes, fm):
    for i, g in enumerate(genes):
        fig, _ = plot_trace(trace, "gamma", n_tune, i, g)
        fig.savefig(outfile + "_trace_gamma_{}_{}.{}".format(i, g, fm))
    plt.close('all')


def _plot_rhat(trace, outfile, genes, gene_conds, fm):
    fig, ax = plot_rhat(trace, "gamma", genes)
    fig.savefig(outfile + "_rhat_gamma." + fm)
    fig, ax = plot_rhat(trace, "beta", gene_conds)
    fig.savefig(outfile + "_rhat_beta." + fm)
    plt.close('all')


def _plot_neff(trace, outfile, genes, gene_cond, fm):
    fig, ax = plot_neff(trace, "gamma", genes)
    fig.savefig(outfile + "_neff_gamma." + fm)
    fig, ax = plot_neff(trace, "beta", gene_cond)
    fig.savefig(outfile + "_neff_beta." + fm)
    plt.close('all')


def _plot_parallel(trace, outfile, ntune, nsample, fm):
    fig, ax = plot_parallel(trace, ntune, nsample)
    fig.savefig(outfile + "_parallel." + fm)
    plt.close('all')


def _plot_hist(trace, outfile, n_tune, genes, fm):
    for i, g in enumerate(genes):
        fig, _ = plot_hist(trace, "gamma", n_tune, i, g)
        fig.savefig(outfile + "_hist_gamma_{}_{}.{}".format(i, g, fm))
    plt.close('all')


def _plot(model, trace, outfile, genes, gene_conds, n_tune, n_sample, model_name):
    graph = model_to_graphviz(model)
    graph.render(filename=outfile + ".dot")

    for fm in ["pdf", "svg", "eps"]:
        _plot_forest(trace, outfile, genes, gene_conds, fm, model_name)
        _plot_trace(trace, outfile, n_tune, genes, fm)
        _plot_hist(trace, outfile, n_tune, genes, fm)
        _plot_neff(trace, outfile, genes, gene_conds, fm)
        _plot_rhat(trace, outfile, genes, gene_conds, fm)
        _plot_parallel(trace, outfile, n_tune, n_sample, fm)


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option("--model-type", type=click.Choice(models), default="shm")
@click.option("--ntune", type=int, default=5000)
@click.option("--nsample", type=int, default=10000)
@click.option("--ninit", type=int, default=100000)
def run(infile, outfile, model_type, ntune, nsample, ninit):
    read_counts = _load_data(infile)

    if model_type == models[0]:
        model, genes, gene_conds = shm(read_counts)
    else:
        model,  genes, gene_conds = flat(read_counts)

    with model:
        trace = pm.sample(nsample, tune=ntune, init="advi", n_init=ninit,
                          chains=4, random_seed=42,
                          discard_tuned_samples=False)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)
    _plot(model, trace, outfile, genes, gene_conds, ntune, nsample, model_type)


if __name__ == "__main__":
    run()
