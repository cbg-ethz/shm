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
from models import shm, shm_independent_l
from models import shm_no_clustering, shm_no_clustering_independent_l


warnings.filterwarnings("ignore")


def _load_data(infile, normalize):
    dat = pd.read_csv(infile, sep="\t")
    if normalize:
        dat["cm"] = sp.mean(dat[["c1", "c2"]].values, axis=1)
        dat["r1"] = sp.log(dat["r1"].values / dat["cm"].values)
        dat["r2"] = sp.log(dat["r2"].values / dat["cm"].values)
    dat = (dat[["Condition", "Gene", "sgRNA", "r1", "r2"]]
           .query("Gene != 'Control'")
           .melt(id_vars=["Gene", "Condition", "sgRNA"],
                 value_vars=["r1", "r2"],
                 var_name="replicate",
                 value_name="counts")
           .sort_values(["Gene", "Condition", "sgRNA", "replicate"])
           )
    dat["sgRNA"] = LabelEncoder().fit_transform(dat["sgRNA"].values)
    if not normalize:
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

    if model == "shm" or model == "shm_independent_l":
        fig, _ = az.plot_forest(trace, var_names="category",
                                credible_interval=0.95)
        _[0].set_title('')
        _[0].set_title('95% credible intervals', size=15, loc="left")
        _[0].spines['left'].set_visible(True)
        _[0].tick_params()
        fig.savefig(outfile + "_forest_category." + fm)
    plt.close('all')


def _plot_trace(trace, outfile, n_tune, keep_burnin, genes, fm):
    for i, g in enumerate(genes):
        fig, _ = plot_trace(trace, "gamma", n_tune, keep_burnin, i, g)
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


def _plot_parallel(trace, outfile, ntune, nsample, keep_burnin, fm):
    fig, ax = plot_parallel(trace, ntune, nsample, keep_burnin)
    fig.savefig(outfile + "_parallel." + fm)
    plt.close('all')


def _plot_hist(trace, outfile, n_tune, keep_burnin, genes, fm):
    for i, g in enumerate(genes):
        fig, _ = plot_hist(trace, "gamma", n_tune, keep_burnin, i, g)
        fig.savefig(outfile + "_hist_gamma_{}_{}.{}".format(i, g, fm))
    plt.close('all')


def _plot(model, trace, outfile, genes, gene_conds, n_tune, n_sample,
          model_name, keep_burnin):
    graph = model_to_graphviz(model)
    graph.render(filename=outfile + ".dot")

    for fm in ["pdf", "svg", "eps"]:
        _plot_forest(trace, outfile, genes, gene_conds, fm, model_name)
        _plot_trace(trace, outfile, n_tune, keep_burnin, genes, fm)
        _plot_hist(trace, outfile, n_tune, keep_burnin,  genes, fm)
        _plot_neff(trace, outfile, genes, gene_conds, fm)
        _plot_rhat(trace, outfile, genes, gene_conds, fm)
        try:
            _plot_parallel(trace, outfile, n_tune, n_sample, keep_burnin, fm)
        except Exception:
            print("Error with some plot")


models = {
    "shm": shm,
    "shm_independent_l": shm_independent_l,
    "shm_no_clustering": shm_no_clustering,
    "shm_no_clustering_independent_l": shm_no_clustering_independent_l
}


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option('--normalize', '-n', is_flag=True)
@click.option('--keep-burnin', '-k', is_flag=True)
@click.option('--filter', '-f', is_flag=True)
@click.option("--model-type", type=click.Choice(models.keys()), default="shm")
@click.option("--ntune", type=int, default=50)
@click.option("--nsample", type=int, default=100)
@click.option("--ninit", type=int, default=1000)
def run(infile, outfile, normalize, keep_burnin, filter,
        model_type, ntune, nsample, ninit):

    read_counts = _load_data(infile, normalize)
    if filter:
        read_counts = read_counts.query("Gene == 'BCR' | Gene == 'PSMB1'")
    model, genes, gene_conds = models[model_type](read_counts, normalize)

    with model:
        trace = pm.sample(nsample, tune=ntune, init="advi", n_init=ninit,
                          chains=4, random_seed=42,
                          discard_tuned_samples=not keep_burnin)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)
    _plot(model, trace, outfile, genes, gene_conds, ntune, nsample,
          model_type, keep_burnin)


if __name__ == "__main__":
    run()
