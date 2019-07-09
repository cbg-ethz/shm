#!/usr/bin/env python3

import os
import pickle

import click
import matplotlib.pyplot as plt
import networkx
import numpy
import pandas as pd
import pymc3 as pm
import seaborn as sns

import shm.plot as sp
from analysis.shlm import SHLM

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


def read_graph(infile):
    with open(infile, "rb") as fh:
        G = pickle.load(fh)
    return G


def _plot_network(graph, data, out_dir):
    plt.figure(figsize=(10, 6))
    pos = networkx.spring_layout(data["graph"])
    networkx.draw_networkx_nodes(
      data["graph"], pos=pos,
      nodelist=list(data['essential_genes']), node_size=300,
      node_color='#316675', font_size=15, alpha=.9, label="Essential genes")
    networkx.draw_networkx_nodes(
      data["graph"].subgraph(["POLR2D", "PSMC1"]), pos=pos,
      nodelist=list(["POLR2D", "PSMC1"]), node_size=300,
      node_color='black', font_size=15, alpha=.9)
    networkx.draw_networkx_nodes(
      data["graph"], pos=pos,
      nodelist=list(data['nonessential_genes']), node_size=300,
      node_color='black', font_size=15, alpha=.9, label="Non-essential genes")
    networkx.draw_networkx_edges(data["graph"], pos=pos)
    plt.axis('off')
    plt.legend(loc='center right', fancybox=False, framealpha=0, shadow=False,
               borderpad=1, bbox_to_anchor=(1, 0), ncol=1)
    plt.savefig(out_dir + "/graph.pdf")
    plt.savefig(out_dir + "/graph.svg")


def _plot_data(data, ppc_trace, out_dir):
    fig, ax = sp.plot_steps(data, bins=30, histtype="bar")
    fig.set_size_inches(6, 3)
    plt.tight_layout()
    plt.savefig(out_dir + "/data.pdf")
    plt.savefig(out_dir + "/data.svg")

    fig, ax = sp.plot_steps(data, ppc_trace, bins=30)
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(out_dir + "/posterior_predictive.pdf")
    plt.savefig(out_dir + "/posterior_predictive.svg")


def _plot_forest(trace, data, model, out_dir):
    genes = data['genes'][list(model._index_to_gene.keys())]
    fig, _ = sp.plot_forest(trace, "gamma", genes)
    plt.savefig(out_dir + "/gamma_forest.pdf")
    plt.savefig(out_dir + "/gamma_forest.svg")
    plt.close('all')


def _plot_trace(trace, model, out_dir):
    fig, ax = sp.plot_neff(trace, "gamma")
    fig.set_size_inches(10, 4)
    plt.savefig(out_dir + "/gamma_neff.svg")
    plt.savefig(out_dir + "/gamma_neff.pdf")
    plt.close('all')

    fig, ax = sp.plot_rhat(trace, "gamma")
    fig.set_size_inches(10, 4)
    plt.savefig(out_dir + "/gamma_rhat.pdf")
    plt.savefig(out_dir + "/gamma_rhat.svg")
    plt.close('all')


def _plot_hist(trace, model, out_dir):
    n_g = trace['gamma'].shape[1]
    for i in range(n_g):
        gene = model._index_to_gene[i]
        fig, ax = sp.plot_hist(trace, "gamma", i, gene)
        fig.set_size_inches(10, 4)
        plt.savefig(out_dir + "/gamma_histogram_{}.svg".format(gene))
        plt.savefig(out_dir + "/gamma_histogram_{}.pdf".format(gene))
        plt.close('all')


def _write_params(model, data, trace, out_dir):
    gamma_true = data['gamma']
    beta_true = data['beta']

    gamma_pred_mean = numpy.mean(trace['gamma'], 0)[
        list(model._index_to_gene.keys())]
    beta_pred_mean = numpy.mean(trace['beta'], 0)[
        list(model._beta_idx_to_gene_cond.keys())]

    pd.DataFrame({"gamma_true": gamma_true,
                  "gamma_inferred_pred": gamma_pred_mean}).to_csv(
      out_dir + "/gamma.tsv", sep="\t", index=False)
    pd.DataFrame({"beta_true": beta_true,
                  "beta_inferred_pred": beta_pred_mean}).to_csv(
      out_dir + "/beta.tsv", sep="\t", index=False)


def _plot_posterior_labels(trace, model, out_dir):
    if 'z' in trace.varnames:
        sns.set(rc={'figure.figsize': (10, 4)})
        ax = sp.plot_posterior_labels(
          trace,
          [model._index_to_gene[x] for x in
           sorted(model._index_to_gene.keys())])
        plt.tight_layout()
        plt.savefig(out_dir + "/posterior_labels.pdf")
        plt.savefig(out_dir + "/posterior_labels.svg")
        plt.close('all')


def plot_model(graph, data, readout, trace, ppc_trace,
               trace_dir, model, out_dir):
    print("params")
    _write_params(model, data, trace, out_dir)
    print("network")
    _plot_network(graph, data, out_dir)
    print("data")
    _plot_data(readout, ppc_trace, out_dir)
    print("trace")
    _plot_trace(trace, model, out_dir)
    #print("hist")
    #_plot_hist(trace, model, out_dir)
    print("forest")
    _plot_forest(trace, data, model, out_dir)
    print("labels")
    _plot_posterior_labels(trace, model, out_dir)


@click.command()
@click.argument('trace', type=str)
@click.argument('readout_file', type=str)
@click.argument('graph_file', type=str)
@click.argument('pickl_file', type=str)
@click.argument('model_type',
                type=click.Choice(["mrf", "clustering", "simple"]))
def run(trace, readout_file, graph_file, pickl_file, model_type):
    out_dir = trace.replace("trace", "results")
    with open(pickl_file, "rb") as fh:
        data = pickle.load(fh)
    readout = pd.read_csv(readout_file, sep="\t")
    graph = read_graph(graph_file)

    with SHLM(readout, model=model_type, graph=graph) as model:
        trace = pm.load_trace(trace, model=model.model)
        ppc_trace = pm.sample_posterior_predictive(trace, 10000, model.model)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plot_model(graph, data, readout, trace, ppc_trace, trace, model, out_dir)


if __name__ == "__main__":
    run()
