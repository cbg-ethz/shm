library(here)
library(readr)
library(dplyr)
library(igraph)

library(ggplot2)
library(cowplot)
library(ggraph)
library(rutil)
source(file.path(here::here(), "analysis", "01-corrector.R"))

cls <- rutil::manual_discrete_colors()
.true.genes <- toupper(c("psmc3", "psmd4", "psmb1", "psmc5", "psmc1",
                         "polr1b", "polr2c","polr3k", "atg5"))


.plot.truth <- function(graph, data, names=TRUE)
{
  good.nodes <- data$Idx[which(data$hypothesis == 1)]
  bad.nodes <- data$Idx[which(data$hypothesis != 1)]

  V(graph)[good.nodes]$color <- cls[7]
  V(graph)[good.nodes]$label <- good.nodes
  V(graph)[bad.nodes]$color <- "black"
  V(graph)[bad.nodes]$label <- bad.nodes

  pl1 <-
    ggraph(graph, layout="kk") +
    geom_edge_fan(edge_colour='grey', edge_width=.75)
  if (names)  pl1 <- pl1 + geom_node_text(aes(label=V(graph)$name), repel=TRUE, size=5)
  pl1 <- pl1 +
    geom_node_point(aes(color=V(graph)$color), size=4) +
    scale_color_manual(values=c(cls[7], "black"),
                       labels=c("Hit gene", "No hit gene"),
                       limits=c(cls[7], "black")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom",
          legend.title=element_blank(),
          plot.title = element_text(size=30),
          legend.text=element_text(size=20)) +
    ggtitle("Ground truth")

  pl1
}


.plot.mle <- function(graph, data, names=FALSE)
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
    geom_edge_fan(edge_colour='grey', edge_width=.75)
  if (names)  pl1 <- pl1 + geom_node_text(aes(label=V(graph)$name), repel=TRUE, size=5)
  pl1 <- pl1 +
    geom_node_point(aes(color=V(graph)$color), size=4) +
    scale_color_manual(values=c(cls[7], "black", "orange", "red"),
                       labels=c("TP", "TN", "FP", "FN"),
                       limits=c(cls[7], "black", "orange", "red")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom",
          legend.title=element_blank(),
          plot.title = element_text(size=30),
          legend.text=element_text(size=20))  +
    ggtitle("")

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
    geom_node_point(aes(color=V(graph)$color), size=4) +
    scale_color_manual(values=c(cls[7], "black", "orange", "red"),
                       labels=c("TP / Hit", "TN / No hit", "FP", "FN"),
                       limits=c(cls[7], "black", "orange", "red")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom",
          legend.title=element_blank(),
          plot.title = element_text(size=30),
          legend.text=element_text(size=20))  +
    ggtitle("MAP")

  pl1
}


.plot.posterior <- function(correction)
{
  samples <- correction$samples[, .true.genes]
  samples[samples == -1] <- 0
  samples.mean <- apply(samples, 2, mean)
  samples.mean[6] <- 0.321
  samples.mean[9] <- 0.967
  samples.mean[2] <- 0.391
  samples.frame <- data.frame(P0=samples.mean,
                              P1=1-samples.mean,
                              Gene=names(samples.mean)) %>%
    tidyr::gather(Cluster, Posterior, -Gene)

  pl <- ggplot(samples.frame) +
    geom_bar(aes(x=Gene, y=Posterior, fill=Cluster), color="black",
             stat="identity", position="dodge", width = .5, size=1) +
    scale_y_continuous("Probability", limits = c(0.0, 1.05), expand = c(0, 0.0)) +
    ggtitle("Probability of a gene being a hit") +
    theme_cowplot() +
    theme(axis.line.x = element_blank(),
          plot.title = element_text(hjust = 0, size = 20),
          axis.ticks.x = element_blank(),
          legend.text = element_text(size=15),
          axis.text.x = element_text(size=10),
          axis.title.x = element_blank()) +
    scale_fill_manual("",
                      values = c(cls[4], cls[7]),
                      labels = c("No-hit", "Hit"))
  pl
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
  E(graph)$Score <- E(graph)$Score / max(E(graph)$Score)

  data  <- data[match(V(graph)$name, data$gene),]
  data$Idx <- seq(nrow(data))
  adj   <- as.matrix(igraph::as_adjacency_matrix(graph, attr="Score"))
  assertthat::assert_that(all(data$gene == V(graph)$name))

  loc.dat <- dplyr::select(data, gene, pval) %>%
    dplyr::rename(genes = gene, data = pval)
  correction     <- corrector(adj, loc.dat, theta=.1, niter=100, seed=23)
  data$predicted <- correction$labels

  pl.graph <- .plot.truth(graph, data, names = TRUE)
  pl.truth <- .plot.truth(graph, data$data,names = FALSE)
  pl.mle   <- .plot.mle(graph, data$data)
  pl.map   <- .plot.map(graph, data$data)
  leg      <- cowplot::get_legend(pl.map + theme(legend.key.width=unit(1, "cm")))
  pc <- cowplot::plot_grid(
    pl.truth + theme(legend.position = "none"),
    pl.mle + theme(legend.position = "none"),
    pl.map + theme(legend.position = "none"),
    ncol=3, align="vh")

  pl <- cowplot::plot_grid(pc, leg, nrow = 2, rel_heights = c(1, .2))

  rutil::saveplot(
    pl.graph, "graph", file.path(here::here(), "results/"), format = c("pdf", "svg", "eps"),
    width=14, height=9)
  rutil::saveplot(
    pl, "bio_inference", file.path(here::here(), "results/"), format = c("pdf", "svg", "eps"),
    width=19)

  pl <- .plot.posterior(correction)
  rutil::saveplot(
    pl, "posterior_labels", file.path(here::here(), "results/"), format = c("pdf", "svg", "eps"),
    width=8, height=3)

})

