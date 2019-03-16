#!/usr/bin/env python

import warnings

import arviz as az
import click
import pandas as pd
import pymc3 as pm
import scipy as sp

from pymc3 import model_to_graphviz
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from plot import plot_trace, plot_neff, plot_rhat, plot_parallel, plot_hist
from models import shm, shm_indendent_l
from models import shm_no_clustering, shm_no_clustering_indendent_l


warnings.filterwarnings("ignore")


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
        fig, _ = az.plot_forest(trace, var_names="category",
                                credible_interval=0.95)
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


def _plot(model, trace, outfile, genes, gene_conds, n_tune, n_sample,
          model_name):
    graph = model_to_graphviz(model)
    graph.render(filename=outfile + ".dot")

    for fm in ["pdf", "svg", "eps"]:
        _plot_forest(trace, outfile, genes, gene_conds, fm, model_name)
        _plot_trace(trace, outfile, n_tune, genes, fm)
        _plot_hist(trace, outfile, n_tune, genes, fm)
        _plot_neff(trace, outfile, genes, gene_conds, fm)
        _plot_rhat(trace, outfile, genes, gene_conds, fm)
        _plot_parallel(trace, outfile, n_tune, n_sample, fm)

models = {
    "shm": shm,
    "shm_indendent_l": shm_indendent_l,
    "shm_no_clustering": shm_no_clustering,
    "shm_no_clustering_indendent_l": shm_no_clustering_indendent_l
}


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option("--model-type", type=click.Choice(models.keys()), default="shm")
@click.option("--ntune", type=int, default=5000)
@click.option("--nsample", type=int, default=10000)
@click.option("--ninit", type=int, default=100000)
def run(infile, outfile, model_type, ntune, nsample, ninit):

    read_counts = _load_data(infile)
    model, genes, gene_conds = models[model_type](read_counts)

    with model:
        trace = pm.sample(nsample, tune=ntune, init="advi", n_init=ninit,
                          chains=4, random_seed=42,
                          discard_tuned_samples=False)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)
    _plot(model, trace, outfile, genes, gene_conds, ntune, nsample, model_type)


if __name__ == "__main__":
    run()
