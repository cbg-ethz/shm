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
from shm.model import Model
from shm.sampler import Sampler

logger = logging.getLogger(__name__)


class HM(ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 family=Family.gaussian,
                 link_function=Link.identity,
                 model=Model.simple,
                 graph=None,
                 node_labels=None,
                 sampler=Sampler.metropolis):

        self.__data = data
        self.__graph = graph
        self.__node_labels = node_labels

        self._set_data()
        self._set_link(link_function)
        self._set_sampler(sampler)
        self._set_family(family)
        self._set_model(model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    @property
    def data(self):
        return self.__data

    @property
    def graph(self):
        return self.__graph

    @property
    def node_labels(self):
        return self.__node_labels

    @property
    def family(self):
        return self.__family

    @property
    def link(self):
        return self.__link_function

    @property
    def sampler(self):
        return self.__sampler

    @property
    def n_genes(self):
        return self.__len_genes

    @property
    def n_conditions(self):
        return self.__len_conds

    @property
    def n_interventions(self):
        return self.__len_intrs

    @property
    def _intervention_data_idx(self):
        return self.__intrs_data_idx

    @property
    def _gene_cond_data_idx(self):
        return self.__gene_cond_data_idx

    @property
    def _beta_idx(self):
        return self.__beta_idx

    @abc.abstractmethod
    def sample(self, n_draw=1000, n_tune=1000, seed=23):
        pass

    def _set_link(self, link_function):
        if isinstance(link_function, str):
            link_function = Link.from_str(link_function)
        self.__link_function = link_function

    def _set_sampler(self, sampler):
        if isinstance(sampler, str):
            sampler = Sampler.from_str(sampler)
        self.__sampler = sampler.value

    def _set_family(self, family):
        if isinstance(family, str):
            family = Family.from_str(family)
        self.__family = family

    def _set_model(self, model):
        if isinstance(model, str):
            model = Model.from_str(model)
        if model == Model.mrf:
            logger.info("Building mrf hierarchical model")
            if not self.__graph:
                raise ValueError("You need to provide a graph")
            self._set_mrf_model()
        elif model == Model.clustering:
            logger.info("Building cluster hierarchical model")
            self._set_clustering_model()
        elif model == Model.simpleimple:
            logger.info("Building eimple hierarchical model")
            self._set_simple_model()
        else:
            raise ValueError("Model not supported")

    @abc.abstractmethod
    def _set_simple_model(self):
        pass

    @abc.abstractmethod
    def _set_clustering_model(self):
        pass

    @abc.abstractmethod
    def _set_mrf_model(self):
        pass

    def _set_data(self):
        data = self.__data
        self.__n, _ = data.shape
        le = LabelEncoder()

        self.__conditions = sp.unique(data[CONDITION].values)
        self.__con_data_idx = le.fit_transform(data[CONDITION].values)
        self.__len_conds = len(self.__conditions)

        self.__genes = sp.unique(data[GENE].values)
        self.__gene_data_idx = le.fit_transform(data[GENE].values)
        self.__len_genes = len(self.__genes)

        self.__intrs = sp.unique(data[INTERVENTION].values)
        self.__intrs_data_idx = le.fit_transform(data[INTERVENTION].values)
        self.__len_intrs = len(self.__intrs)

        self.__beta_idx = le.fit_transform(
          ["{}-{}".format(g, c)
           for g in self.__genes
           for c in self.__conditions])

        self.__gene_cond_data_idx = le.fit_transform(
          ["{}-{}".format(g, c)
           for g, c in zip(data[GENE].values, data[CONDITION].values)])
