import abc
import logging

import numpy as np
from abc import ABC

import pandas as pd
import pymc3 as pm
import scipy as sp
from pymc3.backends import NDArray
from sklearn.preprocessing import LabelEncoder

from shm.family import Family
from shm.globals import INTERVENTION, REPLICATE, GENE, CONDITION
from shm.link import Link
from shm.sampler import Sampler

logger = logging.getLogger(__name__)


class HM(ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 family=Family.gaussian,
                 link=Link.identity,
                 graph=None,
                 node_labels=None,
                 sampler=Sampler.NUTS):

        self.__data = data
        self.__link = link
        self.__family = family
        self.__graph = graph
        self.__node_labels = node_labels

        self.__n, _ = data.shape
        le = LabelEncoder()

        self.__conditions = sp.unique(data[CONDITION].values)
        self.__con_idx = le.fit_transform(data[CONDITION].values)

        self.__genes = sp.unique(data[GENE].values)
        self.__gene_idx = le.fit_transform(data[GENE].values)

        self.__len_genes = len(sp.unique(gene_idx))
        self.__len_conditions = len(sp.unique(con_idx))
        self.__len_sirnas = len(sp.unique(data[INTERVENTION].values))
        self.__len_replicates = len(sp.unique(data[REPLICATE].values))

        self.__beta_idx = sp.repeat(range(len_genes), len_conditions)
        self.__beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

        self.__con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
        self.__gene_conds = ["{}-{}".format(a, b) for a, b in
                             zip(genes[beta_idx], con)]

        self.__l_idx = sp.repeat(
          range(len_genes * len_conditions * len_sirnas_per_gene),
          len_replicates)

        self._set_link(link)
        self._set_sampler(sampler)
        self._set_model()

    def _set_model(self):
        self._set_model()
        self._set_samplers()

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    @property
    def family(self):
        return self.__family

    @abc.abstractmethod
    def sample(self, n_draw=1000, n_tune=1000, seed=23):
        pass

    def sample(self, n_draw=1000, n_tune=1000, seed=23):
        # TODO : add diagnostics
        # TODO: return a multitrace
        np.random.seed(seed)
        trace = NDArray(model=self.__model)
        point = pm.Point(self.__model.test_point, model=self.__model)
        for i in range(n_tune + n_draw):
            point = self._discrete_step(point)
            point, state = self._continuous_step(point)
            trace.record(point, state)

    @property
    def _beta_idx(self):
        return self.__beta_idx

    def _set_link(self, link: Link):
        if link == Link.gaussian:
            setattr(self, link, lambda x: x)
        elif link == Link.log:
            setattr(self, link, numpy.exp)
        else:
            raise ValueError("Incorrect link function specified")

    def _set_sampler(self, sampler):
        if sampler == Sampler.NUTS:
            setattr(self, sampler, pymc3.NUTS)
        elif sampler == Sampler.Metropolis:
            setattr(self, sampler, pymc3.Metropolis)
        else:
            raise ValueError("Incorect link function specified")

    def _set_model(self):
        if self.__graph:
            logger.info("Building mrf hierarchical model")
            self._set_mrf_model()
        else:
            logger.info("Building cluster hierarchical model")
            self._set_clustering_model()
