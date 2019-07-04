#!/usr/bin/env python3
import datetime
import logging
import os
import pickle
import warnings

import click
import numpy
import pandas as pd
import pymc3 as pm

from analysis.copynumber_shlm import CopynumberSHLM
from shm.globals import READOUT, INTERVENTION, CONDITION, GENE, REPLICATE, \
    COPYNUMBER, AFFINITY

warnings.filterwarnings("ignore")
logger = logging.getLogger("pymc3")
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _load_data(infile):
    dat = pd.read_csv(infile, sep="\t")
    cols = [GENE, CONDITION, INTERVENTION, REPLICATE,
            READOUT, COPYNUMBER, AFFINITY]
    for c in cols:
        if c not in dat.columns:
            raise ValueError(
              "Check your column names. Should have: {}".format(c))
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
@click.argument("outfolder", type=str)
@click.option('--model',
              type=click.Choice(["mrf", "clustering"]),
              default="clustering")
@click.option("--ntune", type=int, default=50)
@click.option("--ndraw", type=int, default=100)
@click.option("--nchain", type=int, default=4)
def sample(data_file, outfolder, graph, model, ntune, ndraw, nchain):

    date = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M")
    outfile = os.path.join(outfolder,
                           "biological-{}-{}_model".format(date, model))

    read_counts = _load_data(data_file)
    graph, read_counts = _read_graph(graph, read_counts)

    with CopynumberSHLM(data=read_counts,
                        model=model,
                        graph=graph,
                        use_affinity=True) as model:
        logger.info("Sampling")
        trace = model.sample(draws=ndraw, tune=ntune, chains=nchain, seed=23)

    pm.save_trace(trace, outfile + "_trace", overwrite=True)


if __name__ == "__main__":
    sample()
