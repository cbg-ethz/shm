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
        .sort_values(["Gene", "sgRNA", "replicate"]))
    dat["sgRNA"] = preprocessing.LabelEncoder.fit_transform(dat["sgRNA"].values)
    return dat


def shm(read_counts):
    le = preprocessing.LabelEncoder()

    gene_idx = le.fit_transform(read_counts["Gene"].values)
    len_genes = len(sp.unique(gene_idx))



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
