library(tidyverse)
library(data.table)

string_db <- data.table::fread("../../data_raw/string_db.tsv")
dt <- readr::read_csv("./data_raw/achilles-log_fc.csv")

str_uniq <- unique(dt$string)
cols <- which(string_db$protein1 %in% str_uniq &
              string_db$protein2 %in% str_uniq &
              string_db$combined_score >= 800)
string_db <- string_db[cols]
readr::write_csv(string_db, "./data_raw/string_db.csv")
