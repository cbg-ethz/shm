#!/usr/bin/env python3

import warnings
import arviz as az
import logging
import click
import networkx
import numpy
import pandas as pd
import pymc3 as pm
import scipy as sp
from matplotlib import pyplot as plt
from pymc3 import model_to_graphviz

from shm.family import Family
from shm.globals import GENE
from shm.link import Link
from shm.models.hlm import HLM
from shm.plot import (
    plot_trace, plot_rhat, plot_neff, plot_parallel,
    plot_hist, plot_data, plot_posterior
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("pymc3")
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def _load_data(infile, family):
    dat = pd.read_csv(infile, sep="\t")
    if family != "gaussian":
        dat["readout"] = sp.floor(dat["readout"].values)
    cols = ["gene", "condition", "intervention", "replicate", "readout"]
    for c in cols:
        if c not in dat.columns:
            raise ValueError("Check your column names. Should have: {}".format(c))

    return dat


def _read_graph(infile, data):
    if infile is None:
        return None, data
    genes = numpy.unique(data[GENE].values)
    G = networkx.read_edgelist(
      infile,
      delimiter="\t",
      data=(('weight', float),),
      nodetype=str)
    G = G.subgraph(numpy.sort(genes))
    data = data[data.gene.isin(numpy.sort(G.nodes()))]
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
            logger.error("Error with some plot: {}\n".format(str(e)))


@click.group()
def cli():
    pass


@cli.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option('--family',
              type=click.Choice(["gaussian", "poisson"]),
              default="gaussian")
@click.option('--model',
              type=click.Choice(["mrf", "clustering", "simple"]),
              default="simple")
@click.option("--ntune", type=int, default=50)
@click.option("--ndraw", type=int, default=100)
@click.option("--graph", type=str, default=None)
def sample(infile, outfile, family, model, ntune, ndraw, graph):
    read_counts = _load_data(infile, family)
    link_function = Link.identity if family == "gaussian" else Link.log
    family = Family.gaussian if family == "gaussian" else Family.poisson
    graph, read_counts = _read_graph(graph, read_counts)

    with HLM(data=read_counts,
             family=family,
             link_function=link_function,
             model=model,
             sampler="nuts",
             graph=graph) as model:
        logger.info("Sampling")
        trace = model.sample(draws=ndraw, tune=ntune, chains=4, seed=23)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)


@cli.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option('--family',
              type=click.Choice(["gaussian", "poisson"]),
              default="gaussian")
@click.option('--model',
              type=click.Choice(["mrf", "clustering", "simple"]),
              default="simple")
@click.option("--ntune", type=int, default=50)
@click.option("--ndraw", type=int, default=100)
@click.option("--graph", type=str, default=None)
def plot(infile, outfile, family, model, ntune, ndraw, graph):
    read_counts = _load_data(infile, family)
    link_function = Link.identity if family == "gaussian" else Link.log
    family = Family.gaussian if family == "gaussian" else Family.poisson
    graph, read_counts = _read_graph(graph, read_counts)

    with HLM(data=read_counts,
             family=family,
             link_function=link_function,
             model=model,
             sampler="nuts",
             graph=graph) as model:
        logger.info("Plotting")
        trace = pm.load_trace(outfile + "_trace", model = model.model)
        #plot(model, trace, outfile, genes, gene_conds, ntune, nsample,
        #     model_type, read_counts)

if __name__ == "__main__":
    cli()
