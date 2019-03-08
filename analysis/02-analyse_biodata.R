library(here)
library(readr)
library(dplyr)
library(ggplot2)
library(cowplot)
library(ggraph)
library(igraph)
library(rutil)
source(file.path(here::here(), "analysis", "01-corrector.R"))

cls <- rutil::manual_discrete_colors()


.true.genes <- toupper(c("psmc3", "psmd4", "psmb1", "psmc5", "psmc1",
                         "polr1b", "polr2c","polr3k", "atg5"))


.plot.truth <- function(graph, data)
{
  good.nodes <- data$Idx[which(data$hypothesis == 1)]
  bad.nodes <- data$Idx[which(data$hypothesis != 1)]

  V(graph)[good.nodes]$color <- cls[7]
  V(graph)[good.nodes]$label <- good.nodes
  V(graph)[bad.nodes]$color <- "black"
  V(graph)[bad.nodes]$label <- bad.nodes

  pl1 <-
    ggraph(graph, layout="kk") +
    geom_edge_fan(edge_colour='grey', edge_width=.75) +
    geom_node_text(aes(label=V(graph)$name), repel=TRUE) +
    geom_node_point(aes(color=V(graph)$color), size=4) +
    scale_color_manual(values=c(cls[7], "black"),
                       labels=c("Hit gene", "No hit gene"),
                       limits=c(cls[7], "black")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom", legend.title=element_blank(),
          legend.text=element_text(size=20))

  pl1
}


.plot.mle <- function(graph, data)
{
  good.nodes <- data$Idx[which(data$pval < 0.05 & data$hypothesis == 1)]
  bad.nodes <- data$Idx[which(data$pval >= 0.05 & data$hypothesis == 0)]
  fps <- data$Idx[which(data$pval < 0.05 & data$hypothesis == 0)]
  fns <- data$Idx[which(data$pval >= 0.05 & data$hypothesis == 1)]

  V(graph)[good.nodes]$color <- cls[7]
  V(graph)[good.nodes]$label <- good.nodes
  V(graph)[bad.nodes]$color <- "black"
  V(graph)[bad.nodes]$label <- bad.nodes
  V(graph)[fps]$label <- fps
  V(graph)[fps]$color <- "orange"
  V(graph)[fns]$label <- fns
  V(graph)[fns]$color <- "red"

  pl1 <- ggraph(graph, layout="kk") +
    geom_edge_fan(edge_colour='grey', edge_width=.75) +
    geom_node_text(aes(label=V(graph)$name), repel=TRUE) +
    geom_node_point(aes(color=V(graph)$color), size=4) +
    scale_color_manual(values=c(cls[7], "black", "orange", "red"),
                       labels=c("TP", "TN", "FP", "FN"),
                       limits=c(cls[7], "black", "orange", "red")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom",
          legend.title=element_blank(),
          legend.text=element_text(size=20))

  pl1
}


.plot.map <- function(graph, data)
{

  good.nodes <- data$Idx[which(data$predicted == -1 & data$hypothesis == 1)]
  bad.nodes <- data$Idx[which(data$predicted != -1 & data$hypothesis == 0)]
  fps <- data$Idx[which(data$predicted == -1 & data$hypothesis == 0)]
  fns <- data$Idx[which(data$predicted != -1 & data$hypothesis == 1)]

  V(graph)[good.nodes]$color <- cls[7]
  V(graph)[good.nodes]$label <- good.nodes
  V(graph)[bad.nodes]$color <- "black"
  V(graph)[bad.nodes]$label <- bad.nodes
  V(graph)[fps]$label <- fps
  V(graph)[fps]$color <- "orange"
  V(graph)[fns]$label <- fns
  V(graph)[fns]$color <- "red"

  pl1 <- ggraph(graph, layout="kk") +
    geom_edge_fan(edge_colour='grey', edge_width=.75) +
    geom_node_text(aes(label=V(graph)$name), repel=TRUE) +
    geom_node_point(aes(color=V(graph)$color), size=4) +
    scale_color_manual(values=c(cls[7], "black", "orange", "red"),
                       labels=c("TP", "TN", "FP", "FN"),
                       limits=c(cls[7], "black", "orange", "red")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom",
          legend.title=element_blank(),
          legend.text=element_text(size=20))

  pl1
}


plot.igraph <- . %>%
  igraph::plot.igraph(
    vertex.size=2, vertex.color="black", vertex.label = V(.)$name,
    vertex.label.dist=.5, vertex.label.color="black",
    edge.color="darkgrey", edge.width=.75, edge.arrow.size=.65)



(function() {
  graph <- file.path(here::here(), "data_raw", "mouse_gene_network.tsv") %>%
    readr::read_tsv(col_names = c("G1", "G2", "Score")) %>%
    igraph::graph_from_data_frame(directed=FALSE)
  data  <- file.path(here::here(), "data_raw", "gene_summary.tsv") %>%
    readr::read_tsv() %>%
    dplyr::select(id, `neg|p-value`) %>%
    dplyr::transmute(gene = id, pval = `neg|p-value`, hypothesis = 0)
  data$hypothesis[data$gene %in% .true.genes] <- 1
  idxs <- which(V(graph)$name %in% data$gene)

  graph <- induced_subgraph(graph, idxs)
  data  <- data[match(V(graph)$name, data$gene),]
  data$Idx <- seq(nrow(data))
  adj   <- as.matrix(as.matrix(igraph::as_adjacency_matrix(graph, attr="Score")))
  assertthat::assert_that(all(data$gene == V(graph)$name))

  correction     <- corrector(adj, data$pval, theta=1, niter=10000, seed=23)
  data$predicted <- l[[tab$t[1]]]$labels

  pl.truth <- .plot.truth(graph, data)
  pl.mle <- .plot.mle(graph, data)
  pl.map <- .plot.map(graph, data)
  pc <- cowplot::plot_grid(pl.truth, pl.mle, pl.map, ncol=3, align="vh")
  pc
})

