#!/usr/bin/env bash


function submit_run {
	echo "Submitting jobs"
	for i in "simple" "mrf" "clustering";
    do
        bsub -W 24:00 -n 1 -R "rusage[mem=25000]" python run_shm.py \
            ../../data_raw/biological_ternary-data.tsv \
             ../../data_raw/biological_ternary-graph.pickle \
            "/cluster/home/simondi/simondi/data/shm/biological_ternary-${i}_model" \
            --family gaussian \
            --ntune 20000 \
            --ndraw 30000 \
            --nchain 2 \
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
