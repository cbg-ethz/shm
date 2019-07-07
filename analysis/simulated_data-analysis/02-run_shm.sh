#!/usr/bin/env bash

function submit_run {
    echo "Submitting jobs"
    for i in "mrf" "clustering";
    do
        for j in "0_modified_grnas" "2_modified_grnas" "7_modified_grnas" "10_modified_grnas"
        do
            for k in "noise_sd_low" "noise_sd_middle" "noise_sd_high"
            do
                for f in {1..10}
                do
                    bsub -W 4:00 -n 1 -R "rusage[mem=25000]" python 02-run_shm.py \
                        ${f} \
                        ../../data_raw/simulated-large-${j}-${k}-simulated_data.tsv \
                        ../../data_raw/simulated-large-${j}-${k}-graph.pickle \
                        "/cluster/home/simondi/simondi/data/shm/simulated/simulated_data-${i}_model-${j}-${k}-fold_${f}" \
                        --ntune 30000 \
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
