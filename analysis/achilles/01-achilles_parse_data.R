library(tidyverse)
library(data.table)

sample_info   <- readr::read_csv("../../data_raw/Achilles-sample_info.csv")
replicate_map <- readr::read_csv("../../data_raw/Achilles_replicate_map.csv")
bad_guides    <- readr::read_csv("../../data_raw/Achilles_dropped_guides.csv")
guides_map    <- readr::read_csv("../../data_raw/Achilles_guide_map.csv")
gene2string   <- readr::read_tsv("../../data_raw/gene2string.tsv")
dt <- data.table::fread("../../data_raw/Achilles_logfold_change.csv", sep = ",")


dep_map_ids <- sample_info %>%
  dplyr::rename(n_replicates = Achilles_n_replicates,
                id = DepMap_ID,
                cell_line = stripped_cell_line_name,
                disease_subtype = disease_sutype) %>%
  dplyr::select(id, n_replicates, disease, disease_subtype, cas9_activity,
                cell_line, CCLE_name) %>%
  dplyr::mutate(cas9_activity = as.double(cas9_activity)) %>%
  dplyr::filter(n_replicates >= 4, cas9_activity >= 25) %>%
  dplyr::arrange(desc(cas9_activity)) %>%
  dplyr::group_by(disease) %>%
  dplyr::top_n(1) %>%
  dplyr::ungroup()

replicate_map <- replicate_map %>%
  dplyr::rename(id = DepMap_ID) %>%
  dplyr::filter(id %in% dep_map_ids$id)

guides_map <- guides_map %>%
  dplyr::filter(n_alignments == 1) %>%
  tidyr::separate(gene, c("gene", "id"), " ") %>%
  dplyr::rename(sgRNA = sgrna) %>%
  dplyr::select(sgRNA, gene)

col_idx <- c(1, which(colnames(dt) %in% replicate_map$replicate_ID))
dt <- dt[, ..col_idx] %>%
  tibble::as_tibble() %>%
  dplyr::rename(sgRNA = V1)
dt <- dplyr::left_join(guides_map, dt, by="sgRNA")

gene2string <- gene2string %>%
  dplyr::select(name, string) %>%
  dplyr::rename(gene = name) %>%
  dplyr::filter(gene  %in% dt$gene)
dt <- dplyr::left_join(gene2string, dt, by="gene")

dt <- dt %>%
  tidyr::gather("replicate", "readout", -gene, -string, -sgRNA) %>%
  dplyr::arrange(gene, sgRNA, replicate) %>%
  tidyr::separate(replicate, c("condition", "replicate"), " Rep ") %>%
  tidyr::separate(replicate, c("replicate", "garbage"), " ") %>%
  dplyr::select(gene, condition, sgRNA, string, replicate, readout) %>%
  dplyr::rename(intervention = sgRNA)


readr::write_csv(dep_map_ids, "./data_raw/achilles-sample_info_subset.csv")
readr::write_csv(replicate_map, "./data_raw/achilles-replicate_map_subset.csv")
readr::write_csv(dt, "./data_raw/achilles-log_fc.csv")
