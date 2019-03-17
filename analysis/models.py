import pandas as pd
import pymc3 as pm
import scipy as sp
import theano.tensor as tt
from sklearn.preprocessing import LabelEncoder


def shm(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = sp.unique(read_counts["Condition"].values)
    genes = sp.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))
    len_sirnas = len(sp.unique(read_counts["sgRNA"].values))
    len_replicates = len(sp.unique(read_counts["replicate"].values))
    len_sirnas_per_gene = int(len_sirnas / len_genes)

    beta_idx = sp.repeat(range(len_genes), len_conditions)
    beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    l_idx = sp.repeat(
      range(len_genes * len_conditions * len_sirnas_per_gene), len_replicates)

    with pm.Model() as model:
        p = pm.Dirichlet("p", a=sp.array([1.0, 1.0]), shape=2)
        pm.Potential("p_pot", tt.switch(tt.min(p) < 0.05, -sp.inf, 0))
        category = pm.Categorical("category", p=p, shape=len_genes)

        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        mean_g = pm.Normal("mu_g", mu=sp.array([0, 0]), sd=0.5, shape=2)
        pm.Potential("m_opot", tt.switch(mean_g[1] - mean_g[0] < 0, -sp.inf, 0))
        gamma = pm.Normal("gamma", mean_g[category], tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=len_sirnas)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l[l_idx],
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=len_sirnas)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l[l_idx],
              observed=sp.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds


def shm_independent_l(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = sp.unique(read_counts["Condition"].values)
    genes = sp.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))

    beta_idx = sp.repeat(range(len_genes), len_conditions)
    beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    with pm.Model() as model:
        p = pm.Dirichlet("p", a=sp.array([1.0, 1.0]), shape=2)
        pm.Potential("p_pot", tt.switch(tt.min(p) < 0.05, -sp.inf, 0))
        category = pm.Categorical("category", p=p, shape=len_genes)

        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        mean_g = pm.Normal("mu_g", mu=sp.array([0, 0]), sd=0.5, shape=2)
        pm.Potential("m_opot", tt.switch(mean_g[1] - mean_g[0] < 0, -sp.inf, 0))
        gamma = pm.Normal("gamma", mean_g[category], tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=n)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l,
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=n)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l,
              observed=sp.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds


def shm_no_clustering(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = sp.unique(read_counts["Condition"].values)
    genes = sp.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))
    len_sirnas = len(sp.unique(read_counts["sgRNA"].values))
    len_replicates = len(sp.unique(read_counts["replicate"].values))
    len_sirnas_per_gene = int(len_sirnas / len_genes)

    beta_idx = sp.repeat(range(len_genes), len_conditions)
    beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    l_idx = sp.repeat(
      range(len_genes * len_conditions * len_sirnas_per_gene), len_replicates)

    with pm.Model() as model:
        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        gamma = pm.Normal("gamma", 0, tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=len_sirnas)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l[l_idx],
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=len_sirnas)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l[l_idx],
              observed=sp.squeeze(read_counts["counts"].values))


    return model, genes, gene_conds


def shm_no_clustering_independent_l(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = sp.unique(read_counts["Condition"].values)
    genes = sp.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))

    beta_idx = sp.repeat(range(len_genes), len_conditions)
    beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    with pm.Model() as model:
        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        gamma = pm.Normal("gamma", 0, tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=n)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l,
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=n)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l,
              observed=sp.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds
