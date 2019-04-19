
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
        self.__choice = numpy.random.choice
        self.__point = scipy.stats.bernoulli.rvs(0.5, size=self.__n )

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
        point_label, blanket_labs = point[idx], point[mb]
        mb_weights = self.__adj[mb, idx]
        s1 = numpy.sum((blanket_labs == point_label) * mb_weights)
        s2 = numpy.sum((blanket_labs != point_label) * mb_weights)
        # TODO: is point_label * required (llokn up)
        # i think i can actually leave it out
        return point_label * (s1 - s2)

    def unnormalized_log_prob(self, point):
        print("I am not needed")
        eneg = 0
        for idx, l in enumerate(point):
            # TODO: multiply or not with label here?
            eneg += self._edge_potential(point, idx)
        return eneg

    def random(self, point=None):
        print("I am never needed")
        next_point = numpy.zeros(self.n_nodes)
        if not point:
            point = self.__point
        for idx in range(self.node_labels):
            next_point[idx] = self._gibbs(idx, point)
        self.__point = next_point
        return next_point

    def _gibbs(self, idx, point):
        p = scipy.special.expit(self._edge_potential(point, idx))
        return self.__choice([-1, 1], p=[p, 1 - p])

    def logp(self, value):
        print("I am never needed other than for testing purposes")
        return 0

    def posterior_sample(self, point):
        # TODO
        print("that is where i am at")
        return self.__point

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BinaryMRF}}(\dots)$'.format(name)
