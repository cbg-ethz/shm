#!/usr/bin/env python3
import numpy
import warnings

import arviz as az
import click
import networkx
import pandas as pd
import pymc3 as pm
import scipy as sp

from pymc3 import model_to_graphviz
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

from shm.family import Family
from shm.globals import GENE
from shm.link import Link
from shm.models.hlm import HLM
from shm.plot import (
    plot_trace, plot_rhat, plot_neff, plot_parallel,
    plot_hist, plot_data, plot_posterior
)
from shm.sampler import Sampler

warnings.filterwarnings("ignore")


def _load_data(infile, family):
    dat = pd.read_csv(infile, sep="\t")
    if family == "gaussian":
        print("Taking log-fold change")
        dat["cm"] = sp.mean(dat[["c1", "c2"]].values, axis=1)
        dat["r1"] = sp.log(dat["r1"].values / dat["cm"].values)
        dat["r2"] = sp.log(dat["r2"].values / dat["cm"].values)
    dat = (dat[["Condition", "Gene", "sgRNA", "r1", "r2"]]
           .query("Gene != 'Control'")
           .melt(id_vars=["Gene", "Condition", "sgRNA"],
                 value_vars=["r1", "r2"],
                 var_name="replicate",
                 value_name="counts")
           .sort_values(["Gene", "Condition", "sgRNA", "replicate"]))
    dat["sgRNA"] = LabelEncoder().fit_transform(dat["sgRNA"].values)
    if family != "gaussian":
        dat["counts"] = sp.floor(dat["counts"].values)
    return dat


def _read_graph(infile, data):
    if infile is None:
        return None, data
    genes = data[GENE].values
    G = networkx.read_edgelist(
      infile,
      delimiter="\t",
      data=(('weight', float),),
      nodetype=str)
    G = G.subgraph(numpy.sort(genes))
    data = data[data.id.isin(numpy.sort(G.nodes()))]
    return G, data

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


def _plot_data(data, outfile, fm):
    fig, ax = plot_data(data)
    fig.savefig(outfile + "_data_histogram." + fm)
    plt.close('all')


def _plot_posterior(data, ppc, outfile, fm):
    fig, ax = plot_posterior(data, ppc)
    fig.savefig(outfile + "_data_ppc_histogram." + fm)
    plt.close('all')


def _plot(model, trace, outfile, genes, gene_conds, n_tune, n_sample,
          model_name, keep_burnin, data):
    graph = model_to_graphviz(model)
    graph.render(filename=outfile + ".dot")

    with model:
        ppc = pm.sample_posterior_predictive(trace, 5000, random_seed=23)

    for fm in ["pdf", "svg", "eps"]:
        _plot_data(data, outfile, fm)
        _plot_posterior(data, ppc, outfile, fm)
        _plot_forest(trace, outfile, genes, gene_conds, fm, model_name)
        _plot_trace(trace, outfile, n_tune, keep_burnin, genes, fm)
        _plot_hist(trace, outfile, n_tune, keep_burnin, genes, fm)
        _plot_neff(trace, outfile, genes, gene_conds, fm)
        _plot_rhat(trace, outfile, genes, gene_conds, fm)
        try:
            _plot_parallel(trace, outfile, n_tune, n_sample, keep_burnin, fm)
        except Exception as e:
            print("Error with some plot: {}\n".format(str(e)))


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option('--family',
              type=click.Choice(["gaussian", "poisson"]),
              default="gaussian")
@click.option('--filter', is_flag=True)
@click.option('--model',
              type=click.Choice(["mrf", "clustering", "simple"]),
              default="simple")
@click.option("--sampler",
              type=click.Choice(["nuts", "metropolis"]),
              default="metropolis")
@click.option("--ntune", type=int, default=50)
@click.option("--ndraw", type=int, default=100)
@click.option("--graph", type=str, default=None)
def run(infile, outfile, family, model, filter, sampler, ntune, ndraw, graph):
    read_counts = _load_data(infile, family)
    if filter:
        print("Filtering by genes")
        read_counts = read_counts.query("Gene == 'BCR' | Gene == 'PSMB1'")

    family = Family.gaussian if family == "gaussian" else Family.poisson
    link = Link.identity if family == "gaussian" else Link.log
    graph, read_counts = _read_graph(graph, read_counts)

    with HLM(data=read_counts,
             family=family,
             link=link,
            model=model,
             sampler=sampler,
             graph=graph) as model:
        trace = model.sample(ndraw, ntune, 23)

    # pm.save_trace(trace, outfile + "_trace", overwrite=True)
    # _plot(model, trace, outfile, genes, gene_conds, ntune, nsample,
    #       model_type, keep_burnin, read_counts)


if __name__ == "__main__":
    run()
