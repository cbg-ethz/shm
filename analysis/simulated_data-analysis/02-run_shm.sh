#!/usr/bin/env bash

function submit_run {
	echo "Submitting jobs"
	for i in "mrf" "clustering";
    do
        for j in "0_modified_grnas" "2_modified_grnas" "7_modified_grnas" "10_modified_grnas"
        do
            for k in "noise_sd_0.1" "noise_sd_0.2" "noise_sd_0.5" "noise_sd_1"
            do
                for f in {1..10}
                do
                    echo -W 4:00 -n 1 -R "rusage[mem=25000]" python 02-run_shm.py \
                        ${f} \
                        ../../data_raw/simulated-large-${j}-${k}-simulated_data.tsv \
                        "/cluster/home/simondi/simondi/data/shm/${i}_model-${j}-${k}-fold_${f}" \
                        ../../data_raw/simulated-large-${j}-${k}-graph.pickle \
                        --ntune 20000 \
                        --ndraw 10000 \
                        --model ${i}
                done
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
