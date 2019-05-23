#!/usr/bin/env python3

import pickle

import click
import networkx


def read_graph(infile):
    with open(infile, "rb") as fh:
        G = pickle.load(fh)
    return G


@click.command()
@click.argument('graph_file')
def run(graph_file):
    G = networkx.read_weighted_edgelist(
        graph_file, delimiter="\t", comments="from")
    with open(graph_file.replace(".tsv", ".pickle"), 'wb') as out:
        pickle.dump(G, out)


if __name__ == "__main__":
    run()
