#!/usr/bin/env python3

import datetime
import logging
import os
import pickle
import re
import warnings

import click
import numpy
import pandas as pd
import pymc3
from pymc3 import model_to_graphviz

from analysis.shlm import SHLM
from shm.globals import READOUT, INTERVENTION, CONDITION, GENE, REPLICATE

warnings.filterwarnings("ignore")
logger = logging.getLogger("pymc3")
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _load_data(infile, fold):
    dat = pd.read_csv(infile, sep="\t")
    dat = dat[dat.replicate != "R" + str(fold - 1)]
    cols = [GENE, CONDITION, INTERVENTION, REPLICATE, READOUT]
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
@click.argument("fold", type=int)
@click.argument("data_tsv", type=str)
@click.argument("graph_pickle", type=str)
@click.argument("outfile", type=str)
@click.option('--model',
              type=click.Choice(["mrf", "clustering", "simple"]),
              default="clustering")
@click.option("--ntune", type=int, default=50)
@click.option("--ndraw", type=int, default=100)
@click.option("--nchain", type=int, default=4)
def run(fold, data_tsv, graph_pickle, outfile, model, ntune, ndraw, nchain):

    if fold > 10 or fold < 1:
        raise ValueError("Fold needs to be 1<=x<=10")

    date = datetime.datetime.now().strftime("%Y_%m_%d-%H%M")

    reg = re.match("(.+)/(.+)", outfile)
    file_prefix = reg.group(1) if reg else ""
    file_suffix = reg.group(2) if reg else outfile
    outfile = os.path.join(
      file_prefix, "simulated-{}-{}".format(date, file_suffix))
    logfile = outfile + ".log"

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(logging.Formatter(
      "[%(asctime)s - %(levelname)s - %(name)s]: %(message)s'"))
    logging.getLogger().addHandler(hdlr)

    read_counts = _load_data(data_tsv, fold)
    graph, read_counts = _read_graph(graph_pickle, read_counts)

    with SHLM(data=read_counts, model=model, graph=graph) as model:
        trace = model.sample(draws=ndraw, tune=ntune, chains=nchain, seed=23)

    graphviz = model_to_graphviz(model.model)
    graphviz.render(filename=outfile + "_model", format="png")
    pymc3.save_trace(trace, outfile + "_trace", overwrite=True)


if __name__ == "__main__":
    run()
