library(igraph)
library(scales)
library(ggraph)
library(cowplot)
library(BioNet)
library(dplyr)
library(readr)

n.hits <- 10
n.nohits <- 90

set.seed(23)
pval.nohits <- runif(n.nohits)
pval.hits   <- rbeta(n.hits, .01, 1)
pvals <- c(pval.nohits, pval.hits)
ft$a

set.seed(23)
g <- igraph::random.graph.game(n=n.hits + n.nohits, directed=FALSE, p=.05)
ad <- as_adjacency_matrix(g)
good.nodes <- seq(10)
bad.nodes  <- setdiff(seq(100), good.nodes)
ad[good.nodes, good.nodes] <- rbinom(n=length(ad[good.nodes, good.nodes]), size=1, prob=.5)

g <- graph_from_adjacency_matrix(ad, mode="undirected")
V(g)[good.nodes]$color <- "red"
V(g)[good.nodes]$label <- good.nodes
V(g)[bad.nodes]$color <- "black"
V(g)[bad.nodes]$label <- bad.nodes

pl1 <-
  ggraph(g, layout="kk") +
  geom_edge_fan(edge_colour='grey', edge_width=.75) +
  geom_node_point(aes(color=V(g)$color), size=4) +
  scale_color_manual(values=c("red", "black"), labels=c("Hit gene", "No hit gene"), limits=c("red", "black")) +
  ggraph::theme_graph(base_family="Helvetica") +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text=element_text(size=20))
pl1


df <- data.frame(
  Idx = c(bad.nodes, good.nodes),
  Pvals = pvals)
df$Bin <- cut(df$Pvals, breaks=hist(pvals, plot=F, breaks=10)$breaks, labels=seq(10))
V(g)[df$Idx]$Pvalue <- df$Pvals

pl2 <-
  ggraph(g, layout="kk") +
  geom_edge_fan(edge_colour='grey', edge_width=.75) +
  geom_node_point(aes(color=V(g)$Pvalue), size=4) +
  colorspace::scale_color_continuous_sequential(name="P-value", c1 = 70, c2 = 20, l1 = 100, l2 = 25) +
  ggraph::theme_graph(base_family="Helvetica") +
  theme(legend.position="bottom", legend.title=element_text(size=20), legend.text=element_text(size=15),   legend.key.size = unit(2, "lines"),)
pl2

p <- cowplot::plot_grid(pl1, pl2, align="vh", ncol=2)
p

for (i in c("svg", "eps", "png")) {
  ggsave(filename=paste0("results/graph_random.", i), plot=p, width=12, height=7, dpi=720)
}

MASS::write.matrix(as.matrix(ad), "results/adj_barabasi.tsv", sep="\t")
MASS::write.matrix(df[order(df$Idx),]$Pval, "results/p_values.tsv", sep="\t")
MASS::write.matrix((good.nodes), "results/hit_genes.tsv", sep="\t")
MASS::write.matrix(ft$a, "results/bum_alpha.tsv", sep="\t")
