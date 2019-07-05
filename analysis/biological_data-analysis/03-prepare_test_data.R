library(tidyverse)
library(data.table)
library(igraph)
library(ggraph)


string_db      <- data.table::fread("../../data_raw/achilles/achilles-ppi.csv")
non_essentials <- readr::read_csv("../../data_raw/achilles/achilles-common_nonessentials.csv")
essentials     <- readr::read_csv("../../data_raw/achilles/achilles-common_essentials.csv")
graph_table    <- readr::read_csv("../../data_raw/achilles/achilles-ppi.csv")
dt             <- readr::read_csv("../../data_raw/achilles/achilles-log_fc.csv")


essentials <- bind_rows(tibble(gene=essentials$gene, essential=TRUE),
                        tibble(gene=non_essentials$gene, essential=FALSE)) %>%
  tidyr::separate(gene, c("gene", "entrez"), " ") %>%
  dplyr::select(gene, essential)

cols <- which(graph_table$gene1 %in% essentials$gene &
              graph_table$gene2 %in% essentials$gene &
              string_db$combined_score >= 600)

graph <- graph_table[cols,]
graph <- graph_from_data_frame(graph, FALSE)
graph <- igraph::simplify(graph)
graph <- igraph::induced_subgraph(graph, which(V(graph)$name %in% essentials$gene))

V(graph)$color <- c("black", "red")[
  as.integer(essentials$essential[match(V(graph)$name, essentials$gene)]) + 1
]

graph <- igraph::induced_subgraph(
  graph, unname(which(igraph::degree(graph) > 1)))
graph <- igraph::induced_subgraph(
  graph, which(igraph::components(graph)$membership == 1))

dt <- dt[dt$gene %in% V(graph)$name,]


assertthat::assert_that(all(unique(dt$gene) %in% V(graph)$name))
assertthat::assert_that(all(V(graph)$name %in% unique(dt$gene)))

################################################################################
data.table::fwrite(
  dt,
  "../shm/data_raw/biological-data.tsv", sep="\t")
data.table::fwrite(
  cbind(as_data_frame(graph)[, c(1, 2)], weight=1),
  "../shm/data_raw/biological-graph.tsv", sep="\t")

################################################################################

graph <- igraph::induced_subgraph(graph, c(3, 11, 20))
dt <- dt[which(dt$gene %in% V(graph)$name), ]

assertthat::assert_that(all(unique(dt$gene) %in% V(graph)$name))
assertthat::assert_that(all(V(graph)$name %in% unique(dt$gene)))

data.table::fwrite(
  dt,
  "../shm/data_raw/biological-small_data.tsv", sep="\t")
data.table::fwrite(
  cbind(as_data_frame(graph)[, c(1, 2)], weight=1),
  "../shm/data_raw/biological-small_graph.tsv", sep="\t")

