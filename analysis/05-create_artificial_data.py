#!/usr/bin/env python3

import networkx
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as st
from sklearn import preprocessing
import pickle

gamma_tau = .25
beta_tau = .25
l_tau = .25
data_tau = .25
gamma_tau_non_essential = .1
n_conditions, n_sgrnas, n_replicates = 5, 5, 5

data = pd.read_csv("../data_raw/gene_summary.tsv", sep="\t")
genes = data.id.values
G = networkx.read_edgelist(
  "../data_raw/mouse_gene_network.tsv",
  delimiter="\t",
  data=(('weight', float),),
  nodetype=str)
essential_genes = np.array(list(genes[:4]) + \
                           ["POLR2C", "POLR1B", "PSMC1", "PSMD4", "TH"])

neighbors = []
for c in essential_genes:
    neighbors += networkx.neighbors(G, c)
G = G.subgraph(np.sort(np.unique(neighbors)))

np.random.seed(42)
nonessential_genes = np.random.choice(list(G.nodes), size=30, replace=False)
filter_genes = np.append(essential_genes, nonessential_genes)
G_filtered = G.subgraph(np.sort(filter_genes))


def get_gamma(n_essential, n_nonessential,
              gamma_tau, gamma_tau_non_essential):
    np.random.seed(1)
    gamma_essential = sp.random.normal(-1, scale=gamma_tau, size=n_essential)
    gamma_nonessential = sp.random.normal(0, scale=gamma_tau_non_essential,
                                          size=n_nonessential)
    gamma = sp.append(gamma_essential, gamma_nonessential)
    return gamma, gamma_essential, gamma_nonessential


def build_data(n_essential, n_nonessential, suffix):
    n_genes = n_essential + n_nonessential
    genes = filter_genes[:n_genes]
    gamma, gamma_essential, gamma_nonessential = get_gamma(
      n_essential, n_nonessential, gamma_tau, gamma_tau_non_essential)

    conditions = ["C" + str(i) for i in range(n_conditions)]
    sgrnas = ["S" + str(i) for i in range(n_sgrnas)]
    replicates = ["R" + str(i) for i in range(n_replicates)]

    combinations = [(g, c, s, r)
                    for g in genes
                    for c in conditions
                    for s in sgrnas
                    for r in replicates]

    count_table = pd.DataFrame(
      combinations, columns=["genes", "conditions", "sgrnas", "replicates"])
    sgrna_ids = np.repeat(
      ["S" + str(i) for i in range(n_conditions * n_sgrnas * n_genes)],
      n_replicates)
    count_table.sgrnas = sgrna_ids
    condition_ids = np.repeat(
      ["C" + str(i) for i in range(n_genes * n_conditions)],
      n_sgrnas * n_replicates)
    count_table.conditions = condition_ids

    le = preprocessing.LabelEncoder()
    for i in count_table.columns.values:
        count_table[i] = le.fit_transform(count_table[i])

    beta = st.norm.rvs(np.repeat(gamma, n_conditions), beta_tau)
    l = st.norm.rvs(0, l_tau, size=n_conditions * n_genes * n_sgrnas)
    data = st.norm.rvs(
      l[count_table["sgrnas"]] + beta[count_table["conditions"]],
      data_tau)

    count_table = pd.DataFrame(
      combinations,
      columns=["gene", "condition", "intervention", "replicate"])
    count_table["gamma"] = np.repeat(gamma, count_table.shape[0] / len(gamma))
    count_table["beta"] = np.repeat(beta, count_table.shape[0] / len(beta))
    count_table["l"] = np.repeat(l, count_table.shape[0] / len(l))
    count_table["readout"] = data

    sgrna_ids = np.repeat(
      ["S" + str(i) for i in range(n_conditions * n_sgrnas * n_genes)],
      n_replicates)
    count_table.intervention = sgrna_ids

    count_table.to_csv(
      "../data_raw/easy_simulated_data/{}simulated_data.tsv".format(suffix),
      index=False, sep="\t")

    G_filtered = G.subgraph(genes)
    networkx.readwrite.edgelist.write_weighted_edgelist(
      G_filtered, "../data_raw/easy_simulated_data/{}graph.tsv".format(suffix),
      delimiter="\t")

    data = {
        "graph": G_filtered,
        "essential_genes": essential_genes,
        "nonessential_genes": nonessential_genes,
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

    with open("../data_raw/easy_simulated_data/{}data.pickle".format(suffix),
              "wb") as out:
        pickle.dump(data, out)


def build_large_data():
    n_essential = len(essential_genes)
    n_nonessential = len(nonessential_genes)
    build_data(n_essential, n_nonessential, "")


if __name__ == "__main__":
    build_data(1, 1, "small-")
    #build_large_data()
