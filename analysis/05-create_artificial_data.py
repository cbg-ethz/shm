#!/usr/bin/env python3

import os

import click
import networkx
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as st
from sklearn import preprocessing
import pickle

outpath = os.path.join("..", "data_raw")
gamma_tau = .1
beta_tau = .1
l_tau = .1
data_tau = .1
gamma_tau_non_essential = .1
n_conditions, n_sgrnas, n_replicates = 2, 5, 5

data = pd.read_csv(os.path.join(outpath, "gene_summary.tsv"), sep="\t")
genes = data.id.values
G = networkx.read_edgelist(
  "../data_raw/mouse_gene_network.tsv",
  delimiter="\t",
  data=(('weight', float),),
  nodetype=str)
essential_genes = np.array(list(genes[:4]) +
                           ["POLR2C", "POLR1B", "PSMC1", "PSMD4", "TH"])

neighbors = []
for c in essential_genes:
    neighbors += networkx.neighbors(G, c)
G = G.subgraph(np.sort(np.unique(neighbors)))

np.random.seed(42)
nonessential_genes = np.random.choice(list(G.nodes), size=9, replace=False)
filter_genes = np.append(essential_genes, nonessential_genes)
G_filtered = G.subgraph(np.sort(filter_genes))

A = networkx.subgraph(G_filtered, essential_genes)
B = networkx.subgraph(G_filtered, nonessential_genes)
G_filtered = networkx.Graph()
G_filtered.add_edges_from(list(A.edges()) + list(B.edges()))
G_filtered.add_nodes_from(list(A.nodes()) + list(B.nodes()))


def get_gamma(n_essential, n_nonessential, gamma_tau, gamma_tau_non_essential):
    np.random.seed(1)
    gamma_essential = sp.random.normal(-1, scale=gamma_tau, size=n_essential)
    gamma_nonessential = sp.random.normal(0, scale=gamma_tau_non_essential,
                                          size=n_nonessential)
    gamma = sp.append(gamma_essential, gamma_nonessential)
    return gamma, gamma_essential, gamma_nonessential


def write_file(genes, gamma_essential, gamma_nonessential,
               gamma, beta, l, data, count_table, suffix):
    count_table.to_csv(
      os.path.join(outpath, "easy_simulated_data",
                   "{}simulated_data.tsv".format(suffix)),
      index=False, sep="\t")

    networkx.readwrite.edgelist.write_weighted_edgelist(
      G_filtered,
      os.path.join(outpath, "easy_simulated_data",
                   "{}graph.tsv".format(suffix)),
      delimiter="\t")

    data = {
        "graph": G.subgraph(genes),
        "graph_used": G_filtered,
        "genes": genes,
        "essential_genes": genes[:len(gamma_essential)],
        "nonessential_genes": genes[len(gamma_essential):],
        "gamma_tau": gamma_tau,
        "gamma_tau_non_essential": gamma_tau_non_essential,
        "gamma_essential": gamma_essential,
        "gamma_nonessential": gamma_nonessential,
        "gamma": gamma,
        "beta_tau": beta_tau,
        "beta": beta,
        "l_tau": l_tau,
        "l": l,
        "data_tau": data_tau,
        "data": data,
        "count_table": count_table
    }
    picklepath = os.path.join(outpath, "easy_simulated_data",
                              "{}data.pickle".format(suffix))
    with open(picklepath, "wb") as out:
        pickle.dump(data, out)


def build_data(n_essential, n_nonessential, suffix, with_interventions):
    n_genes = n_essential + n_nonessential
    genes = filter_genes[:n_genes]
    gamma, gamma_essential, gamma_nonessential = get_gamma(
      n_essential, n_nonessential, gamma_tau, gamma_tau_non_essential)
    gamma_dict = {ge: ga for ga, ge in zip(gamma, genes)}

    conditions = ["C" + str(i) for i in range(n_conditions)]
    sgrnas = ["S" + str(i) for i in range(n_sgrnas)]
    replicates = ["R" + str(i) for i in range(n_replicates)]

    combinations = [(g, c, s, r)
                    for g in genes for c in conditions
                    for s in sgrnas for r in replicates]

    count_table = pd.DataFrame(
      combinations, columns=["genes", "conditions", "sgrnas", "replicates"])
    count_table.sgrnas = np.repeat(
      [i for i in range(n_conditions * n_sgrnas * n_genes)], n_replicates)
    condition_ids = np.repeat(
      ["C" + str(i) for i in range(n_conditions)],
      n_sgrnas * n_replicates * n_genes)
    count_table.conditions = condition_ids
    gene_condition_ids = np.array([ "{}-{}".format(g, c)
                                    for g, c in zip(count_table.genes, count_table.conditions)])
    count_table["gamma"] = [gamma_dict[g] for g in count_table["genes"].values]

    conds = np.unique(condition_ids)
    beta = st.norm.rvs(0, beta_tau, size=len(conds))
    beta_dict = {c: b for c, b in zip(conds, beta)}
    count_table["beta"] = [beta_dict[g] for g in count_table["conditions"].values]

    l = st.norm.rvs(0, l_tau, size=n_conditions * n_genes * n_sgrnas)
    if not with_interventions:
        l[:] = 0
    count_table["l"] = l[count_table["sgrnas"]]

    count_table["readout"] = st.norm.rvs(count_table["l"] + \
                       count_table["beta"] + \
                       count_table["gamma"], data_tau)

    write_file(genes, gamma_essential, gamma_nonessential,
               gamma, beta, l, data,
               count_table, suffix)


def build_large_data(with_interventions):
    n_essential = len(essential_genes)
    n_nonessential = len(nonessential_genes)
    build_data(n_essential, n_nonessential, "", with_interventions)


@click.command()
@click.argument('size', type=click.Choice(["small", "large"]))
@click.option("--with-interventions", is_flag=True)
def run(size, with_interventions):
    if size == "small":
        build_data(1, 1, "small-", with_interventions=False)
    else:
        build_large_data(with_interventions)


if __name__ == "__main__":
    run()
