library(tidyverse)
library(data.table)
library(igraph)

string_db <- data.table::fread("../../data_raw/achilles/full_data/string_db.tsv")
dt <- readr::read_csv("../../data_raw/achilles/achilles-log_fc.csv")


str_uniq <- unique(dt$string)
cols <- which(string_db$protein1 %in% str_uniq &
                string_db$protein2 %in% str_uniq &
                string_db$combined_score > 800)
string_db <- string_db[cols]

graph <- graph_from_data_frame(string_db, FALSE)


# '1' created too big a graph, 2 was fine
node.idx <- 2
s <- igraph::induced_subgraph(graph, node.idx)
idx1 <- which(V(graph)$name %in%  igraph::neighbors(graph, node.idx)$name)
idx2 <- which(V(graph)$name %in%  igraph::neighbors(graph, idx1[1])$name)

ind <- igraph::induced_subgraph(graph, c(node.idx, idx1, idx2))
V(ind)$color <- "white"
V(ind)$idx <- seq(length(V(ind)))

n.mod.1 <- which(V(ind)$name %in% igraph::neighbors(ind, 27)$name)
n.mod.2 <- which(V(ind)$name %in% igraph::neighbors(ind, 6)$name)

set.seed(23)
essentials <- c(n.mod.1, sample(n.mod.2, ceiling(length(n.mod.2) * 3 / 4)))

V(ind)$name <- ""
V(ind)$name[essentials] <- paste0("essential_", seq(essentials))
V(ind)$color[essentials] <- rep("blue", length(essentials))

V(ind)$name[c(6, 27)] <- paste0("modified_essential_", 2)
V(ind)$color[c(6, 27)] <- rep("black", 2)

non.essentials <- c(1, setdiff(seq(length(V(ind))), c(essentials, 6, 27)))
V(ind)$name[non.essentials] <- paste0("non_essential_", seq(non.essentials))
V(ind)$color[non.essentials] <- rep("white", 2)


induced.subgraph <- igraph::as_edgelist(ind)

readr::write_csv(as.data.frame(induced.subgraph), "../../data_raw/simulated_graph.csv", col_names=F)
