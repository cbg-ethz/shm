#!/usr/bin/env bash

#bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm --ntune 15000 --nsample 10000 --ninit 10000 ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm/model
#bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm_independent_l --ntune 15000 --nsample 10000 --ninit 10000 ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm_independent/model
#bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm_no_clustering --ntune 15000 --nsample 10000 --ninit 10000 ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm_no_clust/model
#bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm_no_clustering_independent_l --ntune 15000 --nsample 10000 --ninit 10000 ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm_independent_no_clust/model


bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm --ntune 15000 --nsample 10000 --ninit 10000 --normalize \
    ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm_normalize/model
bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm_independent_l --ntune 15000 --nsample 10000 --normalize \
    --ninit 10000 ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm_independent_normalize/model
bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm_no_clustering --ntune 15000 --nsample 10000 --ninit 10000 --normalize \
    ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm_no_clust_normalize/model
bsub -W 24:00 -n 4 -R "rusage[mem=15000]" python shm.py --model-type shm_no_clustering_independent_l --ntune 15000 --nsample 10000 --ninit 10000 --normalize \
    ../data_raw/read_counts-normalized.tsv /cluster/home/simondi/bewi/members/simondi/data/cellline/shm_analysis/shm_independent_no_clust_normalize/model