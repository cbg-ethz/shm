#!/usr/bin/env bash

python run_shm.py \
    ../data_raw/easy_simulated_data/small-simulated_data.tsv \
    ../results/mrf_model_small \
    --family gaussian \
    --ntune 50 \
    --ndraw 50 \
    --model mrf \
    --graph ../data_raw/easy_simulated_data/small-graph.tsv


python run_shm.py \
    ../data_raw/easy_simulated_data/small-simulated_data.tsv \
    ../results/clustering_model_small \
    --family gaussian \
    --ntune 50 \
    --ndraw 50 \
    --model clustering \
    --graph ../data_raw/easy_simulated_data/small-graph.tsv

python run_shm.py \
    ../data_raw/easy_simulated_data/small-simulated_data.tsv \
    ../results/simple_model_small \
    --family gaussian \
    --ntune 50 \
    --ndraw 50 \
    --model simple \
    --graph ../data_raw/easy_simulated_data/small-graph.tsv