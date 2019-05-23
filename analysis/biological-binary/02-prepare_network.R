library(tidyverse)
library(data.table)


string_db <- data.table::fread("../../data_raw/achilles_full/string_db.tsv")
dt <- readr::read_csv("../../data_raw/achilles-log_fc.csv")


str_uniq <- unique(dt$string)
cols <- which(string_db$protein1 %in% str_uniq &
              string_db$protein2 %in% str_uniq &
              string_db$combined_score > 500)
string_db <- string_db[cols]

gene2string <- unique(dt[,c("gene", "string")])

string_db <- string_db %>%
  dplyr::mutate(
    gene1 = gene2string$gene[match(protein1, gene2string$string)],
    gene2 = gene2string$gene[match(protein2, gene2string$string)]) %>%
  dplyr::select(gene1, gene2, protein1:combined_score)

readr::write_csv(string_db, "../../data_raw/achilles-ppi.csv")
