import pickle

import click
import pandas as pd
import pymc3 as pm

from shm.models.hlm import HLM


def read_graph(infile):
    with open(infile, "rb") as fh:
        G = pickle.load(fh)
    return G


@click.command()
@click.argument('mrf_trace', type=str)
@click.argument('clustering_trace', type=str)
@click.argument('readout', type=str)
@click.argument('graph', type=str)
def run(mrf_trace, clustering_trace, readout, graph):
    readout = pd.read_csv(readout, sep="\t")
    G = read_graph(graph)

    with HLM(readout, model="mrf", graph=G) as mrf_model:
        trace_mrf = pm.load_trace(mrf_trace, model=mrf_model.model)
    with HLM(readout, model="clustering", graph=G) as clustering_model:
        trace_c = pm.load_trace(clustering_trace, model=clustering_model.model)

    # df_comp_WAIC = pm.compare(
    #   {mrf_model.model: trace_mrf, clustering_model.model: trace_c})
    # print(df_comp_WAIC)

    df_comp_LOO = pm.compare(
      {mrf_model.model: trace_mrf, clustering_model.model: trace_c}, ic='LOO')
    print(df_comp_LOO)


if __name__ == "__main__":
    run()
