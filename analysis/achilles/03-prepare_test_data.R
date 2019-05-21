library(tidyverse)
library(data.table)
library(igraph)


string_db <- data.table::fread("../../data_raw/achilles-ppi.csv")
non_essentials <- readr::read_csv("../../data_raw/achilles-common_nonessentials.csv")
essentials <- readr::read_csv("../../data_raw/achilles-common_essentials.csv")
graph_table <- readr::read_csv("../../data_raw/achilles-ppi.csv")
dt <- readr::read_csv("../../data_raw/achilles-log_fc.csv")


essentials <- bind_rows(tibble(gene=essentials$gene, essential=TRUE),
                        tibble(gene=non_essentials$gene, essential=FALSE)) %>%
  tidyr::separate(gene, c("gene", "entrez"), " ") %>%
  dplyr::select(gene, essential)

cols <- which(graph_table$gene1 %in% essentials$gene &
              graph_table$gene2 %in% essentials$gene &
              string_db$combined_score >= 600)

graph <- graph_table[cols,]
graph <- graph_from_data_frame(graph, FALSE)
graph <- igraph::induced_subgraph(graph, which(V(graph)$name %in% essentials$gene))

V(graph)$color <- c("black", "red")[
  as.integer(essentials$essential[match(V(graph)$name, essentials$gene)]) + 1
]

graph <- igraph::induced_subgraph(
  graph, unname(which(igraph::degree(graph) > 1)))
graph <- igraph::induced_subgraph(
  graph, which(igraph::components(graph)$membership == 1))

data.table::fwrite(
  dt,
  "../../data_raw/biological_binary-data.csv", sep=",")
data.table::fwrite(
  igraph::as_data_frame(graph),
  "../../data_raw/biological_binary-graph.csv", sep=",")
