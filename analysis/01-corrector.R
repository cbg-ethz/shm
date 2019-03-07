library(LaplacesDemon)
library(igraph)
library(dplyr)


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


corrector <- function(adj, data, theta=1, niter=10000, threshold=1e-5)
{
  set.seed(22)
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
