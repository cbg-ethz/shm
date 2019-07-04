#!/usr/bin/env bash

function run {
    echo "Submitting jobs"
    bsub -W 4:00 -n 1 -R "rusage[mem=25000]" python plot_traces.py \
        ${2} \
        ../../data_raw/biological_ternary-data.tsv \
        "/cluster/home/simondi/simondi/data/shm/achilles/achilles-common_essentials.csv" \
        "/cluster/home/simondi/simondi/data/shm/achilles-common_nonessentials.csv" \
        ../../data_raw/biological-graph.pickle \
        ${1}
}

function usage {
	echo -e "USAGE:\t$0 [clustering|mrf] trace"
	exit
}


if [ $# -eq 0 ]; then
    usage
elif [ $1 == "mrf" ] || [ $1 == "clustering" ]; then
    run $1 $2
else
    usage
fi
