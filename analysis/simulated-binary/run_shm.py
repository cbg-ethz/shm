#!/usr/bin/env python3

import logging
import pickle
import warnings

import click
import numpy
import pandas as pd
import pymc3 as pm
import scipy as sp

from shm.family import Family
from shm.link import Link
from shm.models.shlm import SHLM

warnings.filterwarnings("ignore")
logger = logging.getLogger("pymc3")
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _load_data(infile, family):
    dat = pd.read_csv(infile, sep="\t")
    if family != "gaussian":
        dat["readout"] = sp.floor(dat["readout"].values)
    cols = ["gene", "condition", "intervention", "replicate", "readout"]
    for c in cols:
        if c not in dat.columns:
            raise ValueError("Check your column names. Should have: {}".format(c))

    return dat


def _read_graph(infile, data):
    genes = numpy.unique(data["gene"].values)
    with open(infile, "rb") as fh:
        G = pickle.load(fh)
    G = G.subgraph(numpy.sort(genes))
    data = data[data.gene.isin(numpy.sort(G.nodes()))]
    if len(G.nodes()) != len(numpy.unique(data.gene.values)):
        raise ValueError("Node count different than gene count")
    return G, data


@click.command()
@click.argument("data_file", type=str)
@click.argument("graph", type=str)
@click.argument("outfile", type=str)
@click.option('--family',
              type=click.Choice(["gaussian", "poisson"]),
              default="gaussian")
@click.option('--model',
              type=click.Choice(["mrf", "clustering", "simple"]),
              default="simple")
@click.option("--ntune", type=int, default=50)
@click.option("--ndraw", type=int, default=100)
@click.option("--nchain", type=int, default=4)
def sample(data_file, outfile, graph, family, model, ntune, ndraw, nchain):
    read_counts = _load_data(data_file, family)
    link_function = Link.identity if family == "gaussian" else Link.log
    family = Family.gaussian if family == "gaussian" else Family.poisson
    graph, read_counts = _read_graph(graph, read_counts)

    with SHLM(data=read_counts,
             family=family,
             link_function=link_function,
             model=model,
             sampler="nuts",
             graph=graph) as model:
        logger.info("Sampling")
        trace = model.sample(draws=ndraw, tune=ntune, chains=nchain, seed=23)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)


if __name__ == "__main__":
    sample()
