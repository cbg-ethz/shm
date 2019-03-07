#!/usr/bin/env python3


import click
import logging


def read_mapping_(mapping_file):
    content = {}
    with open(mapping_file, "r") as fh:
        for line in fh:
            if line.startswith("id"):
                continue
            id_, hugo_ = line.strip().split('\t')
            if id_ in content:
            	print("{} already in hash".format(id_))
            content[id_] = hugo_
    return content


def convert(network_file, mapping):
    out = network_file.replace(".tsv", "_mapped.tsv")
    with open(network_file, "r") as fi, open(out, "w") as fo:
        for line in fi:
            g1, g2, score = line.strip().split('\t')
            if g1 in mapping and g2 in mapping:
            	fo.write("{}\t{}\t{}\n".format(mapping[g1], mapping[g2], score))
            elif g1 not in mapping:
                print("G2:{} not found".format(g1))
            elif g2 not in mapping:
                print("G2:{} not found".format(g2))
            else:
            	print("G1:{} and G2:{} not found".format(g1, g2))


@click.command()
@click.argument("network_file", type=str)
@click.argument("mapping_file", type=str)
def run(network_file, mapping_file):
    mapping = read_mapping_(mapping_file)
    convert(network_file, mapping)



if __name__ == "__main__":
    run()
