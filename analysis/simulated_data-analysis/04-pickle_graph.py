#!/usr/bin/env python3

import pickle

import click
import networkx


@click.command()
@click.argument('graph_file')
def run(graph_file):
    G = networkx.read_weighted_edgelist(
        graph_file, delimiter=",", comments="from")
    with open(graph_file.replace(".csv", ".pickle"), 'wb') as out:
        pickle.dump(G, out)


if __name__ == "__main__":
    run()
