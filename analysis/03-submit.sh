#!/usr/bin/env bash


function test_run {
	echo "Running tests"
	bsub -W 4:00 -n 1 -R "rusage[mem=15000]" python run_shm.py sample \
	    ../data_raw/easy_simulated_data/large-simulated_data.tsv \
	    /cluster/home/simondi/simondi/data/shm/test-mrf_model \
	    --family gaussian \
	    --ntune 100 \
	    --ndraw 100 \
	    --model mrf \
	    --graph ../data_raw/easy_simulated_data/large-graph.tsv

	bsub -W 4:00 -n 1 -R "rusage[mem=15000]" python run_shm.py sample \
	    ../data_raw/easy_simulated_data/large-simulated_data.tsv \
	    /cluster/home/simondi/simondi/data/shm/test-clustering_model \
	    --family gaussian \
	    --ntune 100 \
	    --ndraw 100 \
	    --model clustering \
	    --graph ../data_raw/easy_simulated_data/large-graph.tsv

	bsub -W 4:00 -n 1 -R "rusage[mem=15000]" python run_shm.py sample \
	    ../data_raw/easy_simulated_data/large-simulated_data.tsv \
	    /cluster/home/simondi/simondi/data/shm/test-simple_model \
	    --family gaussian \
	    --ntune 100 \
	    --ndraw 100 \
	    --model simple \
	    --graph ../data_raw/easy_simulated_data/large-graph.tsv
}

function submit_run {
	echo "Submitting jobs"
	bsub -W 24:00 -n 1 -R "rusage[mem=25000]" python run_shm.py sample \
	    ../data_raw/easy_simulated_data/large-simulated_data.tsv \
	    /cluster/home/simondi/simondi/data/shm/large-mrf_model \
	    --family gaussian \
	    --ntune 100000 \
	    --ndraw 20000 \
	    --model mrf \
	    --graph ../data_raw/easy_simulated_data/large-graph.tsv

	bsub -W 24:00 -n 1 -R "rusage[mem=25000]" python run_shm.py sample \
	    ../data_raw/easy_simulated_data/large-simulated_data.tsv \
	    /cluster/home/simondi/simondi/data/shm/large-clustering_model \
	    --family gaussian \
	    --ntune 100000 \
	    --ndraw 20000 \
	    --model clustering \
	    --graph ../data_raw/easy_simulated_data/large-graph.tsv

	bsub -W 24:00 -n 1 -R "rusage[mem=25000]" python run_shm.py sample \
	    ../data_raw/easy_simulated_data/large-simulated_data.tsv \
	    /cluster/home/simondi/simondi/data/shm/large-simple_model \
	    --family gaussian \
	    --ntune 100000 \
	    --ndraw 20000 \
	    --model simple \
	    --graph ../data_raw/easy_simulated_data/large-graph.tsv
}


function usage {
	echo -e "USAGE:\t$0 [submit|test]\n"
	exit
}


if [ $# -eq 0 ]; then
   usage
elif [ $1 == "test" ]; then
   test_run
elif [ $1 == "submit" ]; then
   submit_run
fi
