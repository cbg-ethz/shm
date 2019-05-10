#!/usr/bin/env python3

import os
import pickle

import click
import networkx
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st

outpath = os.path.join("..", "data_raw")
gamma_tau = .1
beta_tau = .1
l_tau = .1
data_tau = .1
gamma_tau_non_essential = .1
n_conditions, n_sgrnas, n_replicates = 2, 5, 5


def get_gamma(n_essential, n_nonessential, gamma_tau, gamma_tau_non_essential):
    np.random.seed(1)
    gamma_essential = sp.random.normal(-1, scale=gamma_tau, size=n_essential)
    gamma_nonessential = sp.random.normal(0, scale=gamma_tau_non_essential,
                                          size=n_nonessential)
    gamma = sp.append(gamma_essential, gamma_nonessential)
    return gamma, gamma_essential, gamma_nonessential


def write_file(G, G_filtered, genes, gamma_essential, gamma_nonessential,
               gamma, beta, l, data, count_table, suffix):
    count_table.to_csv(
      os.path.join(outpath, "{}simulated_data.tsv".format(suffix)),
      index=False, sep="\t")

    with open(
      os.path.join(outpath, "{}graph.pickle".format(suffix)), "wb") as out:
        pickle.dump(G_filtered.subgraph(genes), out)

    data = {
        "graph": G.subgraph(genes),
        "graph_used": G_filtered.subgraph(genes),
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
    picklepath = os.path.join(outpath, "{}data.pickle".format(suffix))
    with open(picklepath, "wb") as out:
        pickle.dump(data, out)


def build_data(G, G_filtered, data, essential_genes, nonessential_genes,
               suffix, with_interventions):
    genes = np.append(essential_genes, nonessential_genes)
    n_genes = len(genes)
    n_essential = len(essential_genes)
    n_nonessential = len(nonessential_genes)
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
      combinations, columns=["gene", "condition", "intervention", "replicate"])
    count_table.intervention = np.repeat(
      [i for i in range(n_conditions * n_sgrnas * n_genes)], n_replicates)
    gene_condition_ids = np.array(
      ["{}-{}".format(g, c) for g, c in
       zip(count_table.gene, count_table.condition)])
    count_table["gene_conditions"] = gene_condition_ids
    count_table["gamma"] = [gamma_dict[g] for g in count_table["gene"].values]

    conds = np.unique(gene_condition_ids)
    beta = st.norm.rvs(0, beta_tau, size=len(conds))
    beta_dict = {c: b for c, b in zip(conds, beta)}

    count_table["beta"] = np.array(
      [beta_dict[g] for g in count_table["gene_conditions"].values])
    l = st.norm.rvs(0, l_tau, size=n_conditions * n_genes * n_sgrnas)
    if not with_interventions:
        l[:] = 0
    count_table["l"] = l[count_table["intervention"]]

    count_table["readout"] = st.norm.rvs(
      count_table["l"] + count_table["beta"] + count_table["gamma"], data_tau)
    count_table["intervention"] = ["S" + str(i) for i in
                                   count_table["intervention"]]
    write_file(G, G_filtered, genes, gamma_essential, gamma_nonessential,
               gamma, beta, l, data,
               count_table, suffix)


def filtered_graph(G, essential_genes, nonessential_genes):
    A = networkx.subgraph(G, essential_genes)
    B = networkx.subgraph(G, nonessential_genes)
    G = networkx.Graph()
    G.add_edges_from(list(A.edges()) + list(B.edges()))
    G.add_nodes_from(list(A.nodes()) + list(B.nodes()))
    return G


@click.command()
@click.argument('size', type=click.Choice(["small", "large"]))
@click.option("--with-interventions", is_flag=True)
def run(size, with_interventions):
    data = pd.read_csv(os.path.join(outpath,
                                    "pilot_screen/gene_summary.tsv"), sep="\t")
    genes = data.id.values
    G = networkx.read_edgelist(
      "../data_raw/pilot_screen/mouse_gene_network.tsv",
      delimiter="\t",
      data=(('weight', float),),
      nodetype=str)
    G = G.subgraph(np.unique(genes))
    essential_genes = np.array(
      list(genes[:4]) + ["POLR2C", "POLR1B", "PSMC1", "PSMD4", "TH"])
    nonessential_genes = np.setdiff1d(G.nodes(), essential_genes)

    np.random.seed(42)
    nonessential_genes = np.random.choice(
      nonessential_genes, size=11, replace=False)
    if size == "small":
        essential_genes = np.array(["PSMC5"])
        nonessential_genes = np.array(["PSMB1"])
    filter_genes = np.append(essential_genes, nonessential_genes)
    G = G.subgraph(np.sort(filter_genes))
    G_filtered = G

    if size == "small":
        suffix = "small-"
        G_filtered = G
    else:
        suffix = ""
    build_data(G, G_filtered, data,
               essential_genes, nonessential_genes,
               suffix, with_interventions=False)


if __name__ == "__main__":
    run()
