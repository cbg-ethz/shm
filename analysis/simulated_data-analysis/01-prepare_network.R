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

s1 <- 101
s1 <- which(V(graph)$name %in%  igraph::neighbors(graph, s1)$name)
ind <- igraph::subgraph(graph, sample(s1, 25))
plot(ind)

node.idx <- 23
idx1 <- which(V(graph)$name %in%  igraph::neighbors(graph, node.idx)$name)
idx1
idx2 <- which(V(graph)$name %in%  igraph::neighbors(graph, idx1[1])$name)

ind <- igraph::induced_subgraph(graph, c(node.idx, idx1))
V(ind)$color <- "white"
V(ind)$idx <- seq(length(V(ind)))

plot(ind)

ind <- g
idx.1 <- 9
n.mod.1 <- which(V(ind)$name %in% igraph::neighbors(ind, idx.1)$name)
n.mod.2 <- which(V(ind)$name %in% igraph::neighbors(ind, 6)$name)

set.seed(23)
essentials <- c(n.mod.1, sample(n.mod.2, ceiling(length(n.mod.2) * 3 / 5)))

V(ind)$name <- ""
V(ind)$name[essentials] <- paste0("essential_", seq(essentials))
V(ind)$color[essentials] <- rep("blue", length(essentials))

V(ind)$name[c(6, idx.1)] <- paste0("modified_essential_", 1:2)
V(ind)$color[c(6, idx.1)] <- rep("black", 2)

non.essentials <- c(1, setdiff(seq(length(V(ind))), c(essentials, 6, idx.1)))
V(ind)$name[non.essentials] <- paste0("non_essential_", seq(non.essentials))
V(ind)$color[non.essentials] <- rep("white", length(non.essentials))

plot(ind, vertex.label=V(ind)$idx)


ind <- simplify(ind, remove.multiple = TRUE, remove.loops = TRUE)
induced.subgraph <- igraph::as_edgelist(ind)
induced.subgraph <- unique(as.data.frame(induced.subgraph))
induced.subgraph$weight <- 1
colnames(induced.subgraph)[1:2] <- c("from", "to")

readr::write_csv(induced.subgraph, "../../data_raw/simulated2_graph.csv")

s <- readr::read_csv("../../data_raw/simulated2_graph.csv")


g <- igraph::graph_from_data_frame(s, directed=F)
V(g)$idx <- seq(35)

plot(g)
