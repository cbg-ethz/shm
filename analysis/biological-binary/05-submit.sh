#!/usr/bin/env bash


function submit_run {
	echo "Submitting jobs"
	for i in "simple" "mrf" "clustering";
    do
        bsub -W 24:00 -n 1 -R "rusage[mem=25000]" python run_shm.py sample \
            ../../data_raw/biological_binary-data.tsv \
            "/cluster/home/simondi/simondi/data/shm/biological_binary-${i}_model" \
             ../../data_raw/biological_binary-graph.pickle \
            --family gaussian \
            --ntune 100000 \
            --ndraw 30000 \
            --model ${i}
    done
}

function usage {
	echo -e "USAGE:\t$0 [submit]\n"
	exit
}


if [ $# -eq 0 ]; then
   usage
elif [ $1 == "submit" ]; then
   submit_run
fi
