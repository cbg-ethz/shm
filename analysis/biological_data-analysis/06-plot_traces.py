#!/usr/bin/env python3

import os
import pickle

import click
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import pymc3 as pm
import seaborn as sns
from pymc3 import model_to_graphviz
from sklearn.metrics import confusion_matrix

import shm.plot as sp
from analysis.copynumber_shlm import CopynumberSHLM
from shm.util import compute_posterior_probabilities

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


def _plot_trace(trace, out_dir):
    fig, ax = sp.plot_neff_bar(trace, "gamma")
    fig.set_size_inches(6, 4)
    plt.savefig(out_dir + "/gamma_neff.svg")
    plt.savefig(out_dir + "/gamma_neff.pdf")

    fig, ax = sp.plot_rhat_bar(trace, "gamma")
    fig.set_size_inches(6, 4)
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


def _plot_posterior_labels(trace, model, out_dir):
    gamma_pred_mean = numpy.mean(trace['gamma'], 0)[
        sorted(model._index_to_gene.keys())
    ]

    gamma_pred_mean2 = numpy.mean(trace['gamma'], 0)[list(model._index_to_gene.keys())]

    if 'z' in trace.varnames:
        probs = compute_posterior_probabilities(trace)
        df = pd.DataFrame(
          data=probs,
          columns=["dependency_factor", "neutral", "restriction_factor"])
        df["gene"] = [model._index_to_gene[x] for x in sorted(model._index_to_gene.keys())]
        df["gene2"] = [model._index_to_gene[x] for x in
                      model._index_to_gene.keys()]
        df["gamma"] = gamma_pred_mean
        df["gamma2"] = gamma_pred_mean2
        df = df[["gene", "gene2", "gamma", "gamma2", "dependency_factor", "neutral", "restriction_factor"]]
        df.to_csv(out_dir + "/posterior_labels.tsv", sep="\t", index=False)


def _write_params(model, trace, out_dir):
    gamma_pred_mean = numpy.mean(trace['gamma'], 0)[
        list(model._index_to_gene.keys())]
    beta_pred_mean = numpy.mean(trace['beta'], 0)[
        list(model._beta_idx_to_gene_cond.keys())]

    pd.DataFrame({"gamma_inferred_pred": gamma_pred_mean}).to_csv(
      out_dir + "/gamma.tsv", sep="\t", index=False)
    pd.DataFrame({"beta_inferred_pred": beta_pred_mean}).to_csv(
      out_dir + "/beta.tsv", sep="\t", index=False)


def _plot_confusion_matrix(trace, essentials, non_essentials, model, out_dir):
    probs = compute_posterior_probabilities(trace)
    df = pd.DataFrame(
      data=probs, columns=["dependency_factor", "neutral", "restriction_factor"])
    df["gene"] = [model._index_to_gene[x]
                  for x in sorted(model._index_to_gene.keys())]
    df["is_essential"] = df.dependency_factor + df.restriction_factor
    df["predicted"] = numpy.where(df["neutral"].values > 0.6, 0, 1)
    df["truth"] = -1
    df.loc[df["gene"].isin(essentials["gene"]), "truth"] = 1
    df.loc[df["gene"].isin(non_essentials["gene"]), "truth"] = 0
    df = df[df.truth != -1]

    cm = confusion_matrix(df.truth.values, df.predicted.values)

    fig, ax = sp.plot_confusion_matrix(cm, ["Essential", "Non-essential"])
    fig.set_size_inches(8, 4)
    plt.tight_layout()
    plt.savefig(out_dir + "/confusion_matrix.svg")
    plt.savefig(out_dir + "/confusion_matrix.pdf")
    plt.close('all')


def _plot_graphviz(model, out_dir):
    gra = model_to_graphviz(model)
    gra.render(out_dir + "/graphviz.pdf")


def plot_model(graph, essentials, nonessentials, readout,
               trace, ppc_trace, model, out_dir):

    print("graphviz")
    _plot_graphviz(model.model, out_dir)

    # print("params")
    # _write_params(model, trace, out_dir)
    #
    # print("network")
    # # do we need to plot this? no not really
    # # maybe in the comparison against cluster model
    # # _plot_network(graph, data, out_dir)
    #
    # print("data")
    # _plot_data(readout, ppc_trace, out_dir)
    #
    # print("trace")
    # #_plot_trace(trace, out_dir)
    #
    # #print("hist")
    # # _plot_hist(trace, model, out_dir)
    # # print("forest")
    # # _plot_forest(trace, data, model, out_dir)
    #
    # print("labels")
    # _plot_posterior_labels(trace, model, out_dir)
    #
    # print("confusion")
    # _plot_confusion_matrix(trace, essentials, nonessentials, model, out_dir)


@click.command()
@click.argument('trace', type=str)
@click.argument('readout_tsv', type=str)
@click.argument('essentials_tsv', type=str)
@click.argument('nonessentials_tsv', type=str)
@click.argument('graph_pickle', type=str)
@click.argument('model_type', type=click.Choice(["mrf", "clustering"]))
def run(trace, readout_tsv,
        essentials_tsv, nonessentials_tsv,
        graph_pickle, model_type):
    """
    Plot the TRACE , statistics for biological ternary data and the copynumber
    shlm from the READOUT_FILE used for analysis. Needs
    an ESSENTIALS_FILE/NONESSENTIALS_FILE and the original GRAPH_FILE. Needs
    to provide the MODEL_TYPE used for the analysis.
    """
    out_dir = trace.replace("trace", "results")
    readout = pd.read_csv(readout_tsv, sep="\t")
    graph = read_graph(graph_pickle)
    essentials = pd.read_csv(essentials_tsv, sep=" ")
    nonessentials = pd.read_csv(nonessentials_tsv, sep=" ")

    with CopynumberSHLM(readout,
                        model=model_type,
                        graph=graph,
                        n_states=3,
                        use_affinity=True) as model:
        trace = pm.load_trace(trace, model=model.model)
        # ppc_trace = pm.sample_posterior_predictive(trace, 3000, model.model)
        ppc_trace = None

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plot_model(graph, essentials, nonessentials, readout,
               trace, ppc_trace, model, out_dir)


if __name__ == "__main__":
    run()