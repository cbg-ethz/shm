#!/usr/bin/env bash

function submit_run {
	echo "Submitting jobs"
	for i in "simple" "mrf" "clustering";
    do
        for j in "2_genes_zero" "2_bad_sgrnas" "5_bad_sgrnas" "6_bad_sgrnas" "7_bad_sgrnas"
        do
            bsub -W 24:00 -n 1 -R "rusage[mem=25000]" python run_shm.py sample \
                ../data_raw/${j}-simulated_data.tsv \
                "/cluster/home/simondi/simondi/data/shm/${i}_model-${j}" \
                --family gaussian \
                --ntune 100000 \
                --ndraw 30000 \
                --model ${i} \
                --graph ../data_raw/${j}-graph.pickle
         done
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
