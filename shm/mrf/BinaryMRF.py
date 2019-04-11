
import scipy
import scipy.stats

import networkx
import numpy

from shm.mrf.MRF import MRF


class BinaryMRF(MRF):
    def __init__(self, G, node_labels=None):
        if isinstance(G, networkx.Graph):
            self.__node_labels = numpy.sort(G.nodes())
            self.__adj = networkx.to_numpy_matrix(
              G, nodelist=self.__node_labels, weight='weight')
        elif isinstance(G, numpy.matrix) or isinstance(G, numpy.ndarray):
            if isinstance(G, numpy.matrix):
                G = numpy.asarray(G)
            if node_labels is None:
                raise ValueError("Please provide node labels")
            self.__node_labels = node_labels
            self.__adj = G
        self.__n = len(self.__node_labels)
        self.__binom = scipy.stats.binom.rvs

    @property
    def n_nodes(self):
        return self.__n

    @property
    def node_labels(self):
        return self.__node_labels

    def _markov_blank(self, idx):
        children = numpy.where(self.__adj[idx, :] != 0)[0]
        parents = numpy.where(self.__adj[:, idx] != 0)[0]
        return numpy.unique(numpy.append(children, parents))

    def _edge_potential(self, point, idx):
        mb = self._markov_blank(idx)
        s1 = numpy.sum((point[mb] == point[idx]) * self.__adj[mb, idx])
        s2 = numpy.sum((point[mb] != point[idx]) * self.__adj[mb, idx])
        return s1 - s2

    def unnormalized_log_prob(self, point):
        eneg = 0
        for idx, l in enumerate(point):
            eneg += self._edge_potential(point, idx)
        return eneg

    def sample(self, point):
        next_point = numpy.zeros(self.n_nodes)
        for idx in range(self.node_labels):
            next_point[idx] = self._gibbs(idx, point)
        return next_point

    def _gibbs(self, idx, point):
        p = scipy.special.expit(self._edge_potential(point, idx))
        return self.__binom(1, p)


