# Copyright (C) 2018, 2019 Simon Dirmeier
#
# This file is part of shm.
#
# shm is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# shm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with shm. If not, see <http://www.gnu.org/licenses/>.
#
# @author = 'Simon Dirmeier'
# @email = 'simon.dirmeier@bsse.ethz.ch'

import logging

import scipy
import scipy.stats

import networkx
import numpy
from pymc3 import Discrete


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class CategoricalMRF(Discrete):
    NAME = "CategoricalMRF"

    def __init__(self, G: networkx.Graph, k: int, beta=1, *args, **kwargs):
        self.__node_labels = numpy.sort(G.nodes())
        self.__n = len(self.__node_labels)
        self.__k = k
        self.__adj = networkx.to_numpy_array(
          G, nodelist=self.__node_labels, weight='weight')
        numpy.fill_diagonal(self.__adj, 0)
        super(CategoricalMRF, self).__init__(shape=self.__n, *args, **kwargs)

        self.mode = scipy.repeat(0, self.__n)
        self.__choice = numpy.random.choice
        self.__classes = numpy.arange(k)
        self.__point = self.__choice(self.__classes, size=self.__n)
        self.__blanket = {}

        logger.info("Using MRF edge correction: {}".format(beta))
        self._beta = beta

    @property
    def edge_correction(self):
        return self._beta

    @property
    def name(self):
        return CategoricalMRF.NAME

    @property
    def n_nodes(self):
        return self.__n

    @property
    def _adj(self):
        return self.__adj

    def logp(self, value):
        """Before anyone riots: this is fine, we never need to compute the
        logp. So this is actually never used for critical computations."""
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
        return loglik

    def _gibbs(self, idx, point, node_potentials=None):
        node_pot = self._log_node_potential(node_potentials, idx)
        edge_pot = self._log_edge_potential(point, idx)
        potentials = scipy.exp(edge_pot + node_pot)
        probabilities = potentials / numpy.sum(potentials)
        k = self.__choice(self.__classes, p=probabilities)
        return k

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
        potentials = [
            numpy.sum((blanket_labs == i) * mb_weights)
            for i in self.__classes
        ]
        return self._edge_correction(potentials)

    def _edge_correction(self, val):
        return self._beta * val

    def _markov_blank(self, idx):
        if idx in self.__blanket:
            return self.__blanket[idx]
        children = numpy.where(self.__adj[idx, :] != 0)[0]
        parents = numpy.where(self.__adj[:, idx] != 0)[0]
        blanket = numpy.unique(numpy.append(children, parents))
        self.__blanket[idx] = blanket
        return blanket

    def _loglik(self, gamma, mu, tau):
        ess = [
            scipy.log(scipy.stats.norm.pdf(gamma, mu[i], tau[i]))
            for i in self.__classes
        ]
        return scipy.column_stack(ess)

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{{{}}}'.format(name)
        return r'${} \sim \text{{{}}}(\dots)$'.format(
          name, CategoricalMRF.NAME)
