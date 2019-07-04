#!/usr/bin/env bash

function submit_run {
	echo "Submitting jobs"
	for i in "mrf" "clustering";
    do
        for j in "0_modified_grnas" "2_modified_grnas" "7_modified_grnas" "10_modified_grnas"
        do
            for k in ".1" ".2" ".5" "1"
            do
                bsub -W 4:00 -n 1 -R "rusage[mem=25000]" python 02-run_shm.py sample \
                    ../../data_raw/${j}-simulated_data.tsv \
                    "/cluster/home/simondi/simondi/data/shm/${i}_model-${j}" \
                    ../../data_raw/${j}-graph.pickle \
                    --family gaussian \
                    --ntune 100000 \
                    --ndraw 10000 \
                    --model ${i}
            done
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
