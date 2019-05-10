#!/usr/bin/env python3

import os

import click

import shm
import numpy
import networkx
import matplotlib
import seaborn as sns
import pandas as pd
import pickle
import pymc3 as pm
import logging
import arviz as az
import shm.plot as sp
import matplotlib.pyplot as plt
import seaborn as sns
from shm.models.hlm import HLM

sns.set_style(
  "white",
  {
      "xtick.bottom": True,
      "ytick.left": True,
      "axes.spines.top": False,
      "axes.spines.right": False,
  },
)

cols = ["#E84646", "#316675"]
logger = logging.getLogger(__name__)


def _read_graph(infile, data):
    genes = numpy.unique(data["gene"].values)
    G = networkx.read_edgelist(
      infile,
      delimiter="\t",
      data=(('weight', float),),
      nodetype=str)
    G = G.subgraph(numpy.sort(genes))
    data = data[data.gene.isin(numpy.sort(G.nodes()))]
    return G, data


def _plot_network(graph, data, out_dir, fm):
    plt.figure(figsize=(10, 6))
    pos = networkx.shell_layout(graph)
    networkx.draw_networkx_nodes(
      graph, pos=pos,
      nodelist=data['essential_genes'], node_size=300,
      node_color='#316675', font_size=15, alpha=.9,
      label="Essential gene")
    networkx.draw_networkx_nodes(
      graph, pos=pos,
      nodelist=data['nonessential_genes'], node_size=300,
      node_color='#E84646', font_size=15, alpha=.9,
      label="Non-essential gene")
    networkx.draw_networkx_edges(graph, pos=pos)
    plt.axis('off')
    plt.legend(loc='center right', fancybox=False, framealpha=0, shadow=False,
               borderpad=1, bbox_to_anchor=(1, 0), ncol=1)
    plt.savefig(out_dir + "/graph.{}".format(fm))
    plt.show()


def _plot_data(data, ppc_trace, out_dir, fm):
    fig, ax = sp.plot_steps(data, bins=30, histtype="bar")
    fig.set_size_inches(6, 3)
    plt.tight_layout()
    plt.savefig(out_dir + "/data.{}".format(fm))
    fig, ax = sp.plot_steps(data, ppc_trace, bins=30)
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(out_dir + "/posterior_predictive.{}".format(fm))


def _plot_forest(trace, data, model, out_dir, fm):
    genes = data['genes'][list(model._beta_index_to_gene.keys())]
    fig, _ = sp.plot_forest(trace, "gamma", genes)
    plt.savefig(out_dir + "/gamma_forest.{}".format(fm))

    gene_cond = list(model._beta_idx_to_gene_cond.values())
    fig, _ = sp.plot_forest(trace, "beta", gene_cond)
    plt.savefig(out_dir + "/beta_forest.{}".format(fm))

    if 'z' in trace.varnames:
        fig, _ = sp.plot_forest(trace, "z", gene_cond)
        plt.savefig(out_dir + "/z_forest.{}".format(fm))

    plt.close('all')


def _plot_trace(trace, model, out_dir, fm):
    n_g = trace['gamma'].shape[1]

    fig, ax = sp.plot_neff(trace, "gamma")
    plt.savefig(out_dir + "/gamma_neff.{}".format(fm))
    fig.set_size_inches(10, 4)
    fig, ax = sp.plot_neff(trace, "beta")
    plt.savefig(out_dir + "/beta_neff.{}".format(fm))
    fig.set_size_inches(10, 4)
    plt.close('all')
    fig, ax = sp.plot_rhat(trace, "gamma")
    fig.set_size_inches(10, 4)
    plt.savefig(out_dir + "/gamma_rhat.{}".format(fm))
    fig, ax = sp.plot_rhat(trace, "beta")
    plt.savefig(out_dir + "/beta_rhat.{}".format(fm))
    fig.set_size_inches(10, 4)
    plt.close('all')
    for i in range(n_g):
        g = model._index_to_gene[i]
        fig, _ = sp.plot_trace(trace, "gamma", i, g)
        fig.set_size_inches(10, 4)
        plt.savefig(out_dir + "/gamma_trace_{}_{}.{}".format(i, g, fm))
        plt.close('all')


def _plot_hist(trace, model, out_dir, fm):
    n_g = trace['gamma'].shape[1]
    n_b = trace['beta'].shape[1]
    for i in range(n_g):
        gene = model._index_to_gene[i]
        fig, ax = sp.plot_hist(trace, "gamma", i, gene)
        fig.set_size_inches(10, 4)
        plt.savefig(out_dir + "/gamma_histogram_{}.{}".format(gene, fm))
    for i in range(n_b):
        gene_cond = model._beta_idx_to_gene_cond[i]
        fig, ax = sp.plot_hist(trace, "beta", i, gene_cond)
        fig.set_size_inches(10, 4)
        plt.savefig(out_dir + "/beta_histogram_{}.{}".format(gene_cond, fm))


def _write_params(model, data, trace, out_dir):
    gamma_true = data['gamma']
    beta_true = data['beta']
    gamma_pred_mean = numpy.mean(trace['gamma'], 0)[
        list(model._beta_index_to_gene.keys())]
    beta_pred_mean = numpy.mean(trace['beta'], 0)[
        list(model._beta_idx_to_gene_cond.keys())]

    pd.DataFrame({"gamma_true": gamma_true,
                  "gamma_inferred_pred": gamma_pred_mean}).to_csv(
      out_dir + "/gamma.tsv", sep="\t", index=False)
    pd.DataFrame({"beta_true": beta_true,
                  "beta_inferred_pred": beta_pred_mean}).to_csv(
      out_dir + "/beta.tsv", sep="\t", index=False)


def _plot_posterior_labels(trace, genes, out_dir, fm):
    if 'z' in trace.varnames:
        ax = sp.plot_posterior_labels(trace, genes)
        plt.savefig(out_dir + "/posterior_labels.{}".format(fm))
        plt.close('all')


def plot_model(graph, data, readout, trace, ppc_trace,
               trace_dir, model, out_dir):
    _write_params(model, data, trace, out_dir)
    for fm in ["pdf", "svg"]:
        # _plot_network(graph, data, out_dir, fm)
        # _plot_data(readout, ppc_trace, out_dir, fm)
        # _plot_trace(trace, model, out_dir, fm)
        # _plot_hist(trace, model, out_dir, fm)
        # _plot_forest(trace, data, model, out_dir, fm)
        _plot_posterior_labels(trace, data["genes"], out_dir, fm)


@click.command()
@click.argument('trace', type=str)
@click.argument('readout_file', type=str)
@click.argument('graph_file', type=str)
@click.argument('pickl_file', type=str)
@click.argument('model_type', type=click.Choice(["mrf", "clustering", "simple"]))
def run(trace, readout_file, graph_file, pickl_file, model_type):
    out_dir = trace.replace("trace", "results")
    with open(pickl_file, "rb") as fh:
        data = pickle.load(fh)
    readout = pd.read_csv(readout_file, sep="\t")
    graph, _ = _read_graph(graph_file, readout)

    with HLM(readout, model=model_type, graph=graph) as model:
        trace = pm.load_trace(trace, model=model.model)
        ppc_trace = pm.sample_posterior_predictive(trace, 100, model.model)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    plot_model(graph, data, readout, trace, ppc_trace,
               trace, model, out_dir)


if __name__ == "__main__":
    run()
