
import scipy
import scipy.stats

import networkx
import numpy
from pymc3 import Discrete

from shm.distributions.categorical_mrf import CategoricalMRF
from shm.globals import ESSENTIAL, NON_ESSENTIAL


class BinaryMRF(CategoricalMRF):
    NAME = "BinaryMRF"

    def __init__(self, G: networkx.Graph, *args, **kwargs):
        self.__node_labels = numpy.sort(G.nodes())
        self.__adj = networkx.to_numpy_matrix(
          G, nodelist=self.__node_labels, weight='weight')
        self.__n = len(self.__node_labels)
        super(BinaryMRF, self).__init__(shape=self.__n, *args, **kwargs)

        self.mode = scipy.repeat(NON_ESSENTIAL, self.__n)
        self.__choice = scipy.stats.bernoulli.rvs
        self.__point = scipy.stats.bernoulli.rvs(0.5, size=self.__n)
        self.__blanket = {}

    @property
    def name(self):
        return BinaryMRF.NAME

    @property
    def n_nodes(self):
        return self.__n

    def logp(self, value):
        return 0

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

    def _log_node_potentials(self, gamma, mu, tau):
        loglik = self._loglik(gamma, mu, tau)
        return loglik[:, ESSENTIAL] - loglik[:, NON_ESSENTIAL]

    def _gibbs(self, idx, point, node_potentials=None):
        node_pot = self._log_node_potential(node_potentials, idx)
        edge_pot = self._log_edge_potential(point, idx)
        p = scipy.special.expit(edge_pot + node_pot)
        return self.__choice(p)

    def _log_node_potential(self, node_potentials, idx):
        if node_potentials is None:
            return 0
        return node_potentials[idx]

    def _log_edge_potential(self, point, idx):
        """Parameterization of edge potentials can be taken either from
        1) Murphy - Machine learning
        2) Marin - Bayesian essentials in R
        """
        mb = self._markov_blank(idx)
        point_label, blanket_labs = point[idx], point[mb]
        mb_weights = self.__adj[mb, idx]
        s1 = numpy.sum((blanket_labs == ESSENTIAL) * mb_weights)
        s2 = numpy.sum((blanket_labs != ESSENTIAL) * mb_weights)
        return s1 - s2

    def _markov_blank(self, idx):
        if idx in self.__blanket:
            return self.__blanket[idx]
        children = numpy.where(self.__adj[idx, :] != 0)[0]
        parents = numpy.where(self.__adj[:, idx] != 0)[0]
        blanket = numpy.unique(numpy.append(children, parents))
        self.__blanket[idx] = blanket
        return blanket

    def _loglik(self, gamma, mu, tau):
        if len(tau) == 2:
            tau_0, tau_1 = tau[NON_ESSENTIAL], tau[ESSENTIAL]
        else:
            tau_0, tau_1 = tau, tau
        non = scipy.log2(scipy.stats.norm.pdf(gamma, mu[NON_ESSENTIAL], tau_0))
        ess = scipy.log2(scipy.stats.norm.pdf(gamma, mu[ESSENTIAL], tau_1))
        return scipy.column_stack((non, ess))

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BinaryMRF}}(\dots)$'.format(name)
