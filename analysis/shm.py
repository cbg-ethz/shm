#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore")

import click
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
import theano.tensor as tt

from sklearn import preprocessing
from pymc3 import model_to_graphviz

from matplotlib import pyplot as plt
import seaborn as sns
import arviz as az

sns.set_style(
    "white",
    {
        "xtick.bottom": True,
        "ytick.left": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
)

models = ["shm", "flat"]


def _load_data(infile):
    dat = pd.read_csv(infile, sep="\t")
    dat = (
        dat[["Condition", "Gene", "sgRNA", "r1", "r2"]]
        .query("Gene != 'Control'")
        .melt(
            id_vars=["Gene", "sgRNA"],
            value_vars=["r1", "r2"],
            var_name="replicate",
            value_name="counts")
        .sort_values(["Gene", "Condition", "sgRNA", "replicate"]))
    dat["sgRNA"] = preprocessing.LabelEncoder.fit_transform(dat["sgRNA"].values)
    dat["counts"] = sp.floor(dat["counts"].values)
    return dat


def shm(read_counts: pd.DataFrame):
    n, _ = read_counts.shape
    le = preprocessing.LabelEncoder()

    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))
    len_sirnas = len(sp.unique(read_counts["sgRNA"].values))
    len_replicates = len(sp.unique(read_counts["replicate"].values))
    len_sirnas_per_gene = int(len_sirnas / len_genes)

    beta_idx = np.repeat(range(len_genes), len_conditions)
    beta_data_idx = np.repeat(beta_idx, n / len(beta_idx))

    l_idx = np.repeat(range(len_genes * len_conditions * len_sirnas_per_gene),
                      len_replicates)
    l_idx

    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.array([1., 1.]), shape=2)
        _ = pm.Potential('p_pot',
                                 tt.switch(tt.min(p) < .05, -np.inf, 0))
        category = pm.Categorical('category', p=p, shape=len_genes)

        tau_g = pm.Gamma('tau_g', 1., 1., shape=1)
        mean_g = pm.Normal('mu_g', mu=np.array([0, 0]), sd=.5, shape=2)
        _ = pm.Potential('mop', tt.switch(mean_g[1] - mean_g[0] < 0, -np.inf, 0))
        gamma = pm.Normal('gamma', mean_g[category], tau_g, shape=len_genes)

        tau_b = pm.Gamma('tau_b', 1., 1., shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic('beta', gamma, tau_b, shape=len_genes)
        else:
            beta = pm.Normal('beta', gamma[beta_idx], tau_b, shape=len(beta_idx))
        l = pm.Lognormal('l', 0, .25, shape=len_sirnas)

        pm.Poisson(
          'x',
          mu=np.exp(beta[beta_data_idx]) * l[l_idx],
          observed=sp.squeeze(read_counts["counts"].values))

    return model


def flat(read_counts):
    pass


@click.command()
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option("--model-type", type=click.Choice(models), default="shm")
def run(infile, outfile, model_type):
    read_counts = _load_data(infile)
    if model_type == models[0]:
        model = shm(read_counts)
    else:
        model = flat(read_counts)



if __name__ == "__main__":
    run()
