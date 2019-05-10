import abc
import logging
from abc import ABC

import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder

from shm.family import Family
from shm.globals import INTERVENTION, GENE, CONDITION
from shm.link import Link
from shm.model import Model
from shm.sampler import Sampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class HM(ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 family=Family.gaussian,
                 link_function=Link.identity,
                 model=Model.simple,
                 graph=None,
                 sampler=Sampler.metropolis):
        self.__data = data
        self.__graph = graph
        if graph:
            d_genes = sp.sort(sp.unique(self.__data.gene.values))
            g_genes = sp.sort(graph.nodes())
            if not sp.array_equal(d_genes, g_genes):
                raise ValueError("Graph nodes != data genes")
            self.__node_labels = d_genes

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
    def model_type(self):
        return self.__model_type

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
        self.__model_type = model
        if model == Model.mrf:
            logger.info("Building mrf hierarchical model")
            if not self.__graph:
                raise ValueError("You need to provide a graph")
            self._set_mrf_model()
        elif model == Model.clustering:
            logger.info("Building cluster hierarchical model")
            self._set_clustering_model()
        elif model == Model.simple:
            logger.info("Building simple hierarchical model")
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

    @property
    def _index_to_gene(self):
        return self.__index_to_gene

    @property
    def _index_to_condition(self):
        return self.__index_to_con

    @property
    def _beta_index_to_gene(self):
        return self.__beta_idx_to_gene

    @property
    def _gene_to_beta_index(self):
        return self.__gene_to_beta_idx

    @property
    def _beta_idx_to_gene_cond(self):
        return self.__beta_idx_to_gene_cond

    @property
    def _gene_data_idx(self):
        return self.__gene_data_idx

    @property
    def n_gene_condition(self):
        return self.__len_gene_cond

    # TODO: add methods that map gene to gamma/beta, etc.
    def _set_data(self):
        data = self.__data
        self.__n, _ = data.shape
        le = LabelEncoder()

        self.__gene_data_idx = le.fit_transform(data[GENE].values)
        self.__index_to_gene = {i: e for i, e in zip(
          self.__gene_data_idx, data[GENE].values)}
        self.__genes = sp.unique(list(self.__index_to_gene.values()))
        self.__len_genes = len(self.__genes)

        self.__con_data_idx = le.fit_transform(data[CONDITION].values)
        self.__index_to_con = {i: e for i, e in zip(
          self.__con_data_idx, data[CONDITION].values)}
        self.__conditions = sp.unique(list(self.__index_to_con.values()))
        self.__len_conds = len(self.__conditions)

        self.__intrs_data_idx = le.fit_transform(data[INTERVENTION].values)
        self.__index_to_intervention = {i: e for i, e in zip(
          self.__intrs_data_idx, data[INTERVENTION].values)}
        self.__intrs = sp.unique(data[INTERVENTION].values)
        self.__len_intrs = len(self.__intrs)

        self.__beta_idx = sp.repeat(sp.unique(self.__gene_data_idx),
                                    len(self.__conditions))
        self.__beta_idx_to_gene = {i: self.__index_to_gene[i]
                                   for i in self.__beta_idx}
        self.__gene_to_beta_idx = {e: i for i, e in self.__beta_idx_to_gene.items()}

        self.__gene_cond_data = ["{}-{}".format(g, c)
           for g, c in zip(data[GENE].values, data[CONDITION].values)]
        self.__gene_cond_data_idx = le.fit_transform(self.__gene_cond_data)
        self.__len_gene_cond = len(sp.unique(self.__gene_cond_data))
        self.__beta_idx_to_gene_cond = {
            i: e for i, e in zip(self.__gene_cond_data_idx,
                                 self.__gene_cond_data)
        }
