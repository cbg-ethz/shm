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

readout_file = "../data_raw/easy_simulated_data/small-simulated_data.tsv"
graph_file = "../data_raw/easy_simulated_data/small-graph.tsv"
data_file = "../data_raw/easy_simulated_data/small-data.pickle"


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


def plot_simple_model(trace_dir, out_dir):
    with open(data_file, "rb") as fh:
        data = pickle.load(fh)
    readout = pd.read_csv(readout_file, sep="\t")
    graph, _ = _read_graph(graph_file, readout)

    with HLM(readout) as model:
        trace = pm.load_trace(trace_dir, model=model.model)
        ppc_trace =  pm.sample_posterior_predictive(trace, 10000, model.model)

    gamma_true = data['gamma']
    beta_true = data['beta']
    gamma_pred_mean = numpy.mean(trace['gamma'], 0)[
        list(model._beta_index_to_gene.keys())]
    beta_pred_mean = numpy.mean(trace['beta'], 0)[
        list(model._beta_idx_to_gene_cond.keys())]
    n_g = trace['gamma'].shape[1]
    n_b = trace['beta'].shape[1]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pd.DataFrame({"gamma_true": gamma_true,
                  "gamma_inferred_pred": gamma_pred_mean}).to_csv(
      out_dir + "/gamma.tsv", sep="\t", index=False)
    pd.DataFrame({"beta_true": beta_true,
                  "beta_inferred_pred": beta_pred_mean}).to_csv(
      out_dir + "/beta.tsv", sep="\t", index=False)

    for i in range(n_g):
        gene = model._index_to_gene[i]
        fig, ax = sp.plot_hist(trace, "gamma", i, gene)
        fig.set_size_inches(8, 4)
        plt.savefig(out_dir + "/gamma_{}.svg".format(gene))
        plt.savefig(out_dir + "/gamma_{}.pdf".format(gene))
    for i in range(n_b):
        gene_cond = model._beta_idx_to_gene_cond[i]
        fig, ax = sp.plot_hist(trace, "beta", i, gene_cond)
        fig.set_size_inches(8, 4)
        plt.savefig(out_dir + "/beta_{}.svg".format(gene_cond))
        plt.savefig(out_dir + "/beta_{}.pdf".format(gene_cond))

    fig, ax = sp.plot_steps(readout, ppc_trace,  bins=30)
    fig.set_size_inches(8, 4)
    plt.savefig(out_dir + "/posterior_predictive.svg")
    plt.savefig(out_dir + "/posterior_predictive.pdf")

#
# trace_dir = "../results/small_clustering_model_trace"
# with open(data_file, "rb") as fh:
#     data = pickle.load(fh)
#
# readout = pd.read_csv(readout_file, sep="\t")
# graph, _ = _read_graph(graph_file, readout)
#
# with HLM(readout, model="clustering") as model:
#     trace = pm.load_trace(trace_dir, model=model.model)
#
# sp.plot_hist(trace, "gamma", 0, "");
#
# sp.plot_hist(trace, "gamma", 1, "");
#
# numpy.mean(trace['gamma'], 0)
#
# numpy.mean(trace['z'], 0)
#
# data['beta']
#
# numpy.mean(trace['beta'], 0)[list(model._beta_idx_to_gene_cond.keys())]
#
# # ## Small MRF model
#
#
# trace_dir = "../results/small_mrf_model_trace"
# with open(data_file, "rb") as fh:
#     data = pickle.load(fh)
#
# readout = pd.read_csv(readout_file, sep="\t")
# graph, _ = _read_graph(graph_file, readout)
#
# with HLM(readout, model="mrf", graph=graph) as model:
#     trace = pm.load_trace(trace_dir, model=model.model)
#
# data['gamma']
#
# numpy.mean(trace['gamma'], 0)
#
# sp.plot_hist(trace, "gamma", 0, "");
#
# sp.plot_hist(trace, "gamma", 1, "");
#
# numpy.mean(trace['z'], 0)
#
# numpy.mean(trace['beta'], 0)[list(model._beta_idx_to_gene_cond.keys())]
#
# P1 = numpy.mean(trace['z'], 0)
# P0 = 1 - P1
#
# prob_table = pd.DataFrame(
#   {"p": np.concatenate((P0, P1)),
#    "k": ["No-hit", "No-hit", "Hit", "Hit"],
#    "g": ["G0", "G1", "G0", "G1"]
#    })
#
# cols = sns.color_palette("RdBu", n_colors=7)
# cols = ["#E84646", "#316675"]
#
# # In[38]:
#
#
# ax = sns.barplot(x="g", y="p", hue="k", data=prob_table, palette=cols,
#                  linewidth=2.5, edgecolor=".2");
# sns.despine();
# plt.title('Posterior class label', loc='left', fontsize=16)
# ax.legend(loc='center right', fancybox=False, framealpha=0, shadow=False,
#           borderpad=1,
#           bbox_to_anchor=(1.5, 0.5), ncol=1);
#
# # In[39]:
#
#
# sp.plot_neff(trace, "z");
#
# # In[40]:
#
#
# sp.plot_rhat(trace, "z");
#
# # In[41]:
#
#
# with model.model:
#     ppc_trace = pm.sample_posterior_predictive(trace, 25000, random_seed=1)
#
# # In[42]:
#
#
# sp.plot_steps(readout, ppc_trace, bins=30);


@click.command()
def run():
    plot_simple_model("../results/small_simple_model_trace",
                        "../results/small_simple_model_results")


if __name__ == "__main__":
    run()
