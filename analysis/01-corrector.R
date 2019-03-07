library(LaplacesDemon)
library(igraph)
library(magrittr)
library(ggraph)
library(cowplot)
library(dplyr)

bl <- rutil::manual_discrete_colors()[7]

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
    theme(legend.position="bottom", legend.title=element_text(size=20), legend.text=element_text(size=15),   legend.key.size = unit(2, "lines"),)
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


.sigmoid <- function(x)
{
  1 / (1 + exp(-x))
}


.markov.blanket <- function(i, graph)
{
  children <- which(graph[i, ] == 1)
  parents <- which(graph[, i] == 1)
  unique(c(parents, children))
}


.energy <- function(a, labels, data, adj, theta)
{
  energy <- -log(LaplacesDemon::dinvgamma(a, 5, 1))
  for(i in seq(data))
  {
    if (labels[i] == 1) {
      energy <- energy - log(dbeta(data[i], a, 1))
    } else if (labels[i] == -1) {
      energy <- energy - log(dunif(data[i]))
    } else stop("what you doing, bra")

    mb <- .markov.blanket(i, adj)
    energy <- energy + theta * sum(labels[mb] != labels[i])
  }

  energy
}


.mle <- function(data)
{
  bum.fit <- BioNet::bumOptim(data)
  beta.a <- bum.fit$a
  neg.norm <- dunif(data)
  pos.norm <- dbeta(data, beta.a, 1)
  local.evidence <- matrix(c(neg.norm, pos.norm), ncol=2)
  most.likely <- apply(local.evidence, 1, which.max)
  most.likely
}


.corrector <- function(adj, data, theta=1, niter=10000, threshold=1e-5)
{
  n.obs <- length(data)
  assertthat::assert_that(n.obs == nrow(adj))
  assertthat::assert_that(n.obs == ncol(adj))

  # initialize posterior with MLE
  bum.fit <- BioNet::bumOptim(data)
  beta.a <- bum.fit$a
  neg.norm <- dunif(data)
  pos.norm <- dbeta(data, beta.a, 1)
  local.evidence <- matrix(c(neg.norm, pos.norm), ncol=2)
  most.likely <- apply(local.evidence, 1, which.max)

  labels <- rep(1, length(data))
  labels[most.likely == 1] <- -1
  energy.old <- energy <- Inf

  set.seed(22)
  for (iter in seq(niter)) {
    energy.old <- energy
    a <- LaplacesDemon::rinvgamma(1, 5, 1)
    for (x in seq(n.obs)) {
      mb <- .markov.blanket(x, adj)

      edge.potential <- theta * labels[x] * sum(labels[mb])
      node.potential <- log(dbeta(data[x], a, 1) / dunif(data[x]))
      p  <- .sigmoid(edge.potential - node.potential)

      labels[x] <- ifelse(p >= .5, 1, -1)
    }
    energy <- .energy(a, labels, data, adj, theta)
    if (sum(abs(energy - energy.old)) < threshold) break
  }

  list(labels=labels, energy=energy, a=a)
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
