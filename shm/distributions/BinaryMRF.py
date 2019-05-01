
import scipy
import scipy.stats

import networkx
import numpy
from pymc3 import Discrete


class BinaryMRF(Discrete):
    def __init__(self, G, node_labels=None, *args, **kwargs):
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
        super(BinaryMRF, self).__init__(shape=self.__n, *args, **kwargs)

        self.mode = scipy.repeat(1, self.__n)
        self.__choice = scipy.stats.bernoulli.rvs
        self.__point = scipy.stats.bernoulli.rvs(0.5, size=self.__n)
        self.__blanket = {}

    @property
    def n_nodes(self):
        return self.__n

    @property
    def node_labels(self):
        return self.__node_labels

    def random(self, point=None):
        next_point = numpy.zeros(self.n_nodes)
        for idx in range(self.node_labels):
            next_point[idx] = self._gibbs(idx, point)
        self.__point = next_point
        return next_point

    def posterior_sample(self, z, gamma, mu, tau):
        node_potentials = self._log_node_potentials(gamma, mu, tau)
        next_point = numpy.zeros(self.n_nodes)
        for idx in range(self.n_nodes):
            next_point[idx] = self._gibbs(idx, z, node_potentials)
        self.__point = next_point.astype(numpy.int64)
        return self.__point

    def logp(self, value):
        return 0

    def _gibbs(self, idx, point, node_potentials=None):
        edge_pot = self._log_edge_potential(point, idx)
        node_pot = self._log_node_potential(node_potentials, idx)
        # TODO bug here: this cannot be 2*edge pot
        p = scipy.special.expit(2 * edge_pot + node_pot)
        return self.__choice(p)

    def _log_edge_potential(self, point, idx):
        mb = self._markov_blank(idx)
        point_label, blanket_labs = point[idx], point[mb]
        mb_weights = self.__adj[mb, idx]
        s1 = numpy.sum((blanket_labs == point_label) * mb_weights)
        s2 = numpy.sum((blanket_labs != point_label) * mb_weights)
        return s1 - s2

    def _markov_blank(self, idx):
        if idx in self.__blanket:
            return self.__blanket[idx]
        children = numpy.where(self.__adj[idx, :] != 0)[0]
        parents = numpy.where(self.__adj[:, idx] != 0)[0]
        blanket = numpy.unique(numpy.append(children, parents))
        self.__blanket[idx] = blanket
        return blanket

    def _log_node_potential(self, node_potentials, idx):
        if node_potentials is None:
            return 0
        return node_potentials[idx]

    def _log_node_potentials(self, gamma, mu, tau):
        loglik = self._loglik(gamma, mu, tau)
        return loglik[:, 1] - loglik[:, 0]

    def _loglik(self, gamma, mu, tau):
        neg = scipy.log2(scipy.stats.norm.pdf(gamma, mu[0], tau[0]))
        pos = scipy.log2(scipy.stats.norm.pdf(gamma, mu[1], tau[1]))
        return scipy.column_stack((neg, pos))

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BinaryMRF}}(\dots)$'.format(name)
