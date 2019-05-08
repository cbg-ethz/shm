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
    networkx.draw(graph, labels={e: e for e in data['essential_genes']},
                  node_size=300, font_size=15, alpha=.9,
                  pos=networkx.shell_layout(graph));
    plt.savefig(out_dir + "/graph.{}".format(fm)


def _plot_data(data, ppc_trace, out_dir, fm):
    fig, ax = sp.plot_steps(readout, bins=30)
    fig.set_size_inches(8, 4)
    plt.savefig(out_dir + "/data.{}".format(fm))
    fig, ax = sp.plot_steps(readout, ppc_trace, bins=30)
    fig.set_size_inches(8, 4)
    plt.savefig(out_dir + "/posterior_predictive.{}".format(fm))


def _plot_forest(trace, data, model, out_dir, fm):
    fig, _ = az.plot_forest(trace, var_names="gamma", credible_interval=0.95)
    _[0].set_title('')
    _[0].set_title('95% credible intervals', size=15, loc="left")
    _[0].spines['left'].set_visible(True)
    _[0].set_yticklabels(genes)
    _[0].tick_params()
    plt.savefig(outfile + "_forest_gamma." + fm)

    fig, _ = az.plot_forest(trace, var_names="beta", credible_interval=0.95)
    _[0].set_title('')
    _[0].set_title('95% credible intervals', size=15, loc="left")
    _[0].spines['left'].set_visible(True)
    _[0].set_yticklabels(gene_cond)
    _[0].tick_params()
    plt.savefig(outfile + "_forest_beta." + fm)

    if 'z' in trace.varnames:
        fig, _ = az.plot_forest(trace, var_names="z", credible_interval=0.95)
        _[0].set_title('')
        _[0].set_title('95% credible intervals', size=15, loc="left")
        _[0].spines['left'].set_visible(True)
        _[0].tick_params()
        fig.savefig(outfile + "_forest_category." + fm)

    plt.close('all')


def _plot_trace():
    n_g = trace['gamma'].shape[1]
    n_b = trace['beta'].shape[1]

    fig, ax = plot_neff(trace, "gamma")
    plt.savefig(out_dir + "/gamma_neff.svg")
    plt.savefig(out_dir + "/gamma_neff.pdf")
    fig, ax = plot_neff(trace, "beta")
    plt.savefig(out_dir + "/beta_neff.svg")
    plt.savefig(out_dir + "/beta_neff.pdf")

    fig, ax = plot_rhat(trace, "gamma")
    plt.savefig(out_dir + "/gamma_rhat.svg")
    plt.savefig(out_dir + "/gamma_rhat.pdf")
    fig, ax = plot_rhat(trace, "beta")
    plt.savefig(out_dir + "/beta_rhat.svg")
    plt.savefig(out_dir + "/beta_rhat.pdf")

    for i in range(n_g):
        g = model._index_to_gene[i]
        fig, _ = plot_trace(trace, "gamma", i, g)
        plt.savefig(out_dir + "/gamma_trace_{}_{}.eps".format(i, g))
        plt.savefig(out_dir + "/gamma_trace_{}_{}.pdf".format(i, g))
    for i in range(n_b):
        g = model._beta_idx_to_gene_cond[i]
        fig, _ = plot_trace(trace, "beta", i, g)
        plt.savefig(out_dir + "/beta_trace_{}_{}.eps".format(i, g))
        plt.savefig(out_dir + "/beta_trace_{}_{}.pdf".format(i, g))


def _plot_hist(n_g, n_b, model, trace, out_dir, fm):
    n_g = trace['gamma'].shape[1]
    n_b = trace['beta'].shape[1]
    for i in range(n_g):
        gene = model._index_to_gene[i]
        fig, ax = sp.plot_hist(trace, "gamma", i, gene)
        fig.set_size_inches(8, 4)
        plt.savefig(out_dir + "/gamma_{}.{}".format(gene, fm))
    for i in range(n_b):
        gene_cond = model._beta_idx_to_gene_cond[i]
        fig, ax = sp.plot_hist(trace, "beta", i, gene_cond)
        fig.set_size_inches(8, 4)
        plt.savefig(out_dir + "/beta_{}.{}".format(gene_cond, fm))


def _write_params():
    gamma_true = data['gamma']
    beta_true = data['beta']
    gamma_pred_mean = numpy.mean(trace['gamma'], 0)[
        list(model._beta_index_to_gene.keys())]
    beta_pred_mean = numpy.mean(trace['beta'], 0)[
        list(model._beta_idx_to_gene_cond.keys())]
    n_g = trace['gamma'].shape[1]
    n_b = trace['beta'].shape[1]

    pd.DataFrame({"gamma_true": gamma_true,
                  "gamma_inferred_pred": gamma_pred_mean}).to_csv(
      out_dir + "/gamma.tsv", sep="\t", index=False)
    pd.DataFrame({"beta_true": beta_true,
                  "beta_inferred_pred": beta_pred_mean}).to_csv(
      out_dir + "/beta.tsv", sep="\t", index=False)


def _plot_posterior_labels(trace, genes, out_dir, fm):
    P1 = numpy.mean(trace['z'], 0)
    P0 = 1 - P1

    prob_table = pd.DataFrame({
        "Probability": np.concatenate((P0, P1)),
        "o": np.concatenate(np.repeat("No-hit", len(z)),
                            np.repeat("Hit", len(z))),
        "Gene": np.tile(genes, 2) }
    )

    cols = ["#E84646", "#316675"]

    ax = sns.barplot(x="Gene", y="Probability", hue="o",
                     data=prob_table, palette=cols,
                     linewidth=2.5, edgecolor=".2");
    sns.despine();
    plt.title('Posterior class label', loc='left', fontsize=16)
    ax.legend(loc='center right', fancybox=False, framealpha=0, shadow=False,
              borderpad=1, bbox_to_anchor=(1.5, 0.5), ncol=1);

    plt.figure(out_dir + "/posterior_labels.{}".format(fm))


def plot_model(graph, data, readout, trace, ppc_trace, trace_dir, out_dir):
    for fm in ["pdf", "svg"]:
        _plot_network(graph, data, out_dir, fm)
        _plot_data(data, ppc_trace, out_dir, fm)
        _plot_forest(trace, data, model.model, out_dir, fm)
        _plot_trace(trace, outfile, n_tune, keep_burnin, genes, fm)
        _plot_hist(trace, outfile, n_tune, keep_burnin, genes, fm)
        _plot_posterior_labels(trace, data["genes"], out_dir, fm)
        except Exception as e:
            logger.error("Error with some plot: {}\n".format(str(e)))


@click.command()
@click.argument('folder', type=str)
@click.argument('readout_file', type=str)
@click.argument('graph_file', type=str)
@click.argument('pickl_file', type=str)
@click.argument('model', type=click.Choice(["mrf", "clustering", "simple"]))
def run(folder, readout_file, graph_file, pickl_file, model):

    with open(pickl_file, "rb") as fh:
        data = pickle.load(fh)
    readout = pd.read_csv(readout_file, sep="\t")
    graph, _ = _read_graph(graph_file, readout)

    with HLM(readout, model=model, graph=graph) as model:
        trace     = pm.load_trace(trace_dir, model=model.model)
        ppc_trace = pm.sample_posterior_predictive(trace, 10000, model.model)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    plot_model(graph, data, readout, trace, ppc_trace,
               folder, folder.replace("trace", "results"))


if __name__ == "__main__":
    run()
