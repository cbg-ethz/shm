
import networkx
import numpy
import scipy
import scipy.stats

from shm.distributions.categorical_mrf import CategoricalMRF
from shm.globals import ESSENTIAL, NON_ESSENTIAL


class BinaryMRF(CategoricalMRF):
    NAME = "BinaryMRF"

    def __init__(self, G: networkx.Graph, *args, **kwargs):
        super(BinaryMRF, self).__init__(G=G, k=2, *args, **kwargs)

        self.mode = scipy.repeat(NON_ESSENTIAL, self.n_nodes)
        self.__choice = scipy.stats.bernoulli.rvs
        self.__classes = numpy.arange(2)
        self.__point = scipy.stats.bernoulli.rvs(0.5, size=self.n_nodes)
        self.__blanket = {}

    @property
    def name(self):
        return BinaryMRF.NAME

    def _log_node_potentials(self, gamma, mu, tau):
        loglik = self._loglik(gamma, mu, tau)
        return loglik[:, 1] - loglik[:, 0]

    def _gibbs(self, idx, point, node_potentials=None):
        node_pot = self._log_node_potential(node_potentials, idx)
        edge_pot = self._log_edge_potential(point, idx)
        p = scipy.special.expit(edge_pot + node_pot)
        return self.__choice(p)

    def _log_edge_potential(self, point, idx):
        """Parameterization of edge potentials can be taken either from
        1) Murphy - Machine learning
        2) Marin - Bayesian essentials in R
        """
        mb = self._markov_blank(idx)
        point_label, blanket_labs = point[idx], point[mb]
        mb_weights = self._adj[mb, idx]
        s1 = numpy.sum((blanket_labs == 1) * mb_weights)
        s2 = numpy.sum((blanket_labs != 1) * mb_weights)
        return s1 - s2

    def _markov_blank(self, idx):
        if idx in self.__blanket:
            return self.__blanket[idx]
        children = numpy.where(self._adj[idx, :] != 0)[0]
        parents = numpy.where(self._adj[:, idx] != 0)[0]
        blanket = numpy.unique(numpy.append(children, parents))
        self.__blanket[idx] = blanket
        return blanket

    def _loglik(self, gamma, mu, tau):
        if len(tau) == 2:
            tau_0, tau_1 = tau[0], tau[1]
        else:
            tau_0, tau_1 = tau, tau
        a0 = scipy.log(scipy.stats.norm.pdf(gamma, mu[0], tau_0))
        a1 = scipy.log(scipy.stats.norm.pdf(gamma, mu[1], tau_1))
        return scipy.column_stack((a0, a1))

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BinaryMRF}}(\dots)$'.format(name)
