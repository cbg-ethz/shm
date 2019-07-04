#!/usr/bin/env bash

function run {
    python 06-plot_traces.py \
    ../../../../results/biological_ternary/current/biological_ternary-${1}_model_trace/ \
    ../../data_raw/biological_ternary-data.tsv \
    ../../../../data_raw/achilles/achilles-common_essentials.csv \
    ../../../../data_raw/achilles/achilles-common_nonessentials.csv \
    ../../data_raw/biological_ternary-graph.pickle \
    ${1}
}

function usage {
	echo -e "USAGE:\t$0 [clustering|mrf]"
	exit
}


if [ $# -eq 0 ]; then
    usage
elif [ $1 == "mrf" ] || [ $1 == "clustering" ]; then
    run $1
else
    usage
fi
