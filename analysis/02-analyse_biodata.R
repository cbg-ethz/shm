library(ggplot2)
library(cowplot)
library(ggraph)
library(igraph)


.plot <- function(adj, pvals)
{
  g <- graph_from_adjacency_matrix(adj, mode="undirected")
  good.nodes <- pvals$Idx[which(pvals$Hypothesis == 1)]
  bad.nodes <- pvals$Idx[which(pvals$Hypothesis != 1)]

  V(g)[good.nodes]$color <- bl
  V(g)[good.nodes]$label <- good.nodes
  V(g)[bad.nodes]$color <- "black"
  V(g)[bad.nodes]$label <- bad.nodes

  pl1 <-
    ggraph(g, layout="kk") +
    geom_edge_fan(edge_colour='grey', edge_width=.75) +
    geom_node_point(aes(color=V(g)$color), size=4) +
    scale_color_manual(values=c(bl, "black"), labels=c("Hit gene", "No hit gene"), limits=c(bl, "black")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom", legend.title=element_blank(), legend.text=element_text(size=20))

  V(g)[pvals$Idx]$Pvalue <- pvals$Pvals
  pl2 <-
    ggraph(g, layout="kk") +
    geom_edge_fan(edge_colour='grey', edge_width=.75) +
    geom_node_point(aes(color=V(g)$Pvalue), size=4) +
    colorspace::scale_color_continuous_sequential(name="P-value", c1 = 70, c2 = 20, l1 = 100, l2 = 25) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom",
          legend.title=element_text(size=20),
          legend.text=element_text(size=15),
          legend.key.size = unit(2, "lines"),)
  pl2

  list(pl1, pl2)
}

.plot.prediction <- function(adj, pvals)
{

  pl <- .plot(adj, pvals)

  g <- graph_from_adjacency_matrix(adj, mode="undirected")
  good.nodes <- pvals$Idx[which(pvals$predicted == -1 & pvals$Hypothesis == 1)]
  bad.nodes <- pvals$Idx[which(pvals$predicted != -1 & pvals$Hypothesis == 0)]
  fps <- pvals$Idx[which(pvals$predicted == -1 & pvals$Hypothesis == 0)]
  fns <- pvals$Idx[which(pvals$predicted != -1 & pvals$Hypothesis == 1)]
  V(g)[good.nodes]$color <- bl
  V(g)[good.nodes]$label <- good.nodes
  V(g)[bad.nodes]$color <- "black"
  V(g)[bad.nodes]$label <- bad.nodes
  V(g)[fps]$label <- fps
  V(g)[fps]$color <- "orange"
  V(g)[fns]$label <- fps
  V(g)[fns]$color <- "red"
  s <- pvals$Idx[which(pvals$Hypothesis == 1)]
  V(g)[49]$color <- "red"
  V(g)[88]$color <- "black"

  ggraph(g, layout="kk") +
    geom_edge_fan(edge_colour='grey', edge_width=.75) +
    geom_node_point(aes(color=V(g)$color), size=4) +
    scale_color_manual(values=c(bl, "black", "orange", "red"),
                       labels=c("TP", "TN", "FP", "FN"),
                       limits=c(bl, "black", "orange", "red")) +
    ggraph::theme_graph(base_family="Helvetica") +
    theme(legend.position="bottom", legend.title=element_blank(), legend.text=element_text(size=20))

  list(pl[[1]], pl[[2]], pl1)
}


here <- paste0(here::here(), "/src/corrector/")
here.res <- paste0(here, "/results/validation.rda")
here.out <- paste0(here, "/results/")
validation <- load(here.res)

p <- .plot(adj, pvals)
pc <- cowplot::plot_grid(plotlist=p, ncol=2, align="vh")

for (i in c("svg", "png")) {
  rutil::saveplot(pc, "scale_free_benchmark_data", out.folders=here.out, width=12, height=6)
}


pvals$rank <- seq(nrow(pvals))
pvals <- pvals %>% arrange(Idx)

correction <- .corrector(adj, pvals$Pvals)
pvals$mle <- .mle(pvals$Pvals)
pvals$predicted <- correction$labels
pvals <- pvals %>% arrange(rank)

p <- .plot.prediction(adj, pvals)
pc <- cowplot::plot_grid(plotlist=p, ncol=3, align="vh")
pc
for (i in c("svg", "png"))
{
  rutil::saveplot(pc, "scale_free_benchmark_data_ted", format=i, out.folders=here.out, width=15, height=6)
}


bl <- rutil::manual_discrete_colors()[7]

