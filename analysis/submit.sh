#!/usr/bin/env bash


function test_run {
	bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python run_shm.py \
	    ../data_raw/easy_simulated_data/small-simulated_data.tsv \
	    ../results/mrf_model_small \
	    --family gaussian \
	    --ntune 50 \
	    --ndraw 50 \
	    --model mrf \
	    --graph ../data_raw/easy_simulated_data/small-graph.tsv


	bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python run_shm.py \
	    ../data_raw/easy_simulated_data/small-simulated_data.tsv \
	    ../results/clustering_model_small \
	    --family gaussian \
	    --ntune 50 \
	    --ndraw 50 \
	    --model clustering \
	    --graph ../data_raw/easy_simulated_data/small-graph.tsv

	bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python run_shm.py \
	    ../data_raw/easy_simulated_data/small-simulated_data.tsv \
	    ../results/simple_model_small \
	    --family gaussian \
	    --ntune 50 \
	    --ndraw 50 \
	    --model simple \
	    --graph ../data_raw/easy_simulated_data/small-graph.tsv
}


function submit_run {
	python run_shm.py \
	    ../data_raw/easy_simulated_data/simulated_data.tsv \
	    ../results/mrf_model_small \
	    --family gaussian \
	    --ntune 50 \
	    --ndraw 50 \
	    --model mrf \
	    --graph ../data_raw/easy_simulated_data/graph.tsv


	python run_shm.py \
	    ../data_raw/easy_simulated_data/simulated_data.tsv \
	    ../results/clustering_model_small \
	    --family gaussian \
	    --ntune 50 \
	    --ndraw 50 \
	    --model clustering \
	    --graph ../data_raw/easy_simulated_data/graph.tsv

	python run_shm.py \
	    ../data_raw/easy_simulated_data/simulated_data.tsv \
	    ../results/simple_model_small \
	    --family gaussian \
	    --ntune 50 \
	    --ndraw 50 \
	    --model simple \
	    --graph ../data_raw/easy_simulated_data/graph.tsv
}


function usage {
	echo -e "USAGE:\t$0 [submit|test]\n"
	exit
}


if [ $# -eq 0 ]; then
   usage
elif [ $1 == "test" ]; then
   echo "t"
elif [ $1 == "submit" ]; then
   echo "su"
fi