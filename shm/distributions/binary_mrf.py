
import networkx
import numpy
import scipy
import scipy.stats

from shm.distributions.categorical_mrf import CategoricalMRF


class BinaryMRF(CategoricalMRF):
    NAME = "BinaryMRF"

    def __init__(self, G: networkx.Graph, beta=1, *args, **kwargs):
        super(BinaryMRF, self).__init__(G=G, k=2, beta=beta, *args, **kwargs)

        self.mode = scipy.repeat(0, self.n_nodes)
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
        s0 = numpy.sum((blanket_labs == 0) * mb_weights)
        return self._edge_correction(s1 - s0)

    def _loglik(self, gamma, mu, tau):
        a0 = scipy.log(scipy.stats.norm.pdf(gamma, mu[0], tau[0]))
        a1 = scipy.log(scipy.stats.norm.pdf(gamma, mu[1], tau[1]))
        return scipy.column_stack((a0, a1))

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BinaryMRF}}(\dots)$'.format(name)
