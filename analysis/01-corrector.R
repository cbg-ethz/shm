library(LaplacesDemon)
library(igraph)
library(dplyr)

.sigmoid <- function(x)
{
  1 / (1 + exp(-x))
}


.markov.blanket <- function(i, graph)
{
  children <- which(graph[i, ] != 0)
  parents <- which(graph[, i] != 0)
  unique(c(parents, children))
}


.edge.potential <- function(labels, x, adj)
{
  mb <- .markov.blanket(x, adj)

  sum((labels[mb] == labels[x]) * adj[mb, x]) -
    sum((labels[mb] != labels[x]) * adj[mb, x])
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

    energy <- energy + theta * .edge.potential(labels, i, adj)
  }

  energy
}


.mle <- function(data)
{
  bum.fit <- BioNet::bumOptim(data$data, labels = data$genes)
  beta.a <- bum.fit$a
  neg.norm <- dunif(data$data)
  pos.norm <- dbeta(data$data, beta.a, 1)
  local.evidence <- matrix(c(neg.norm, pos.norm), ncol=2)
  most.likely <- apply(local.evidence, 1, which.max)

  list(most.likely=most.likely,
       beta.a=beta.a,
       neg.norm=neg.norm,
       pos.norm=pos.norm,
       local.evidence=local.evidence)
}



.map.corrector <- function(adj, data, theta=1, niter=10000,
                           nwarm=1000, threshold=1e-5, seed=22, hyper.a=5, hyper.b=1)
{
  set.seed(seed)
  assertthat::assert_that("genes" %in% colnames(data))
  assertthat::assert_that("data" %in% colnames(data))

  n.obs <- length(data$data)
  assertthat::assert_that(n.obs == nrow(adj))
  assertthat::assert_that(n.obs == ncol(adj))

  # initialize posterior with MLE
  mle <- .mle(data)
  beta.a <- mle$beta.a
  most.likely <- mle$most.likely

  labels <- rep(1, length(data$data))
  labels[most.likely == 1] <- -1
  energy.old <- energy <- Inf

  samples <- matrix(0, niter, n.obs,
                    dimnames = list(NULL, data$genes))

  for (iter in seq(niter + nwarm))
  {
    if (iter %% 1000 == 0) cat(paste0("Iter: ", iter, "\n"))
    energy.old <- energy
    labels.old <- labels
    #a <- LaplacesDemon::rinvgamma(1, 5, 1)
    a <- beta.a
    for (x in seq(n.obs))
    {
      edge.potential <- theta * labels[x] * .edge.potential(labels, x, adj)
      node.potential <- log(dbeta(data$data[x], a, 1) / dunif(data$data[x]))
      p  <- .sigmoid(edge.potential - node.potential)
      #labels[x] <- ifelse(p >= .5, 1, -1)
      labels[x] <- sample(c(1, -1), size = 1, prob = c(p, 1 - p))
    }

    if (iter > nwarm) {
      samples[iter - nwarm,] <- labels
    }

    energy <- .energy(a, labels, data$data, adj, theta)
    if (energy < energy.old) {
      labels <- labels.old
    }
  }

  list(labels=labels, energy=energy, a=a, samples=samples)
}


corrector <- function(adj, data, take.map=TRUE, ...)
{
  if (take.map) ret <- .map.corrector(adj, data, ...)
  else stop("Not implemented yet!")
  ret
}
