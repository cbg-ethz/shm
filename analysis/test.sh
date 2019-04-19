python run_shm.py \
/home/simon/PROJECTS/shm/data_raw/read_counts_parsed-normalized.tsv \
/home/simon/PROJECTS/shm/results/hm/test \
--family gaussian
--filter \
--sampler nuts 
--ntune 50
--ndraw 50
--model simple
--graph /home/simon/PROJECTS/shm/data_raw/mouse_gene_network.tsv