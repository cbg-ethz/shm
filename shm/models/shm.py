import abc
import logging
from abc import ABC

import pandas as pd
import pymc3 as pm
import scipy as sp
from sklearn.preprocessing import LabelEncoder
import theano.tensor as tt

from shm.distributions.binary_mrf import BinaryMRF
from shm.distributions.categorical_mrf import CategoricalMRF
from shm.family import Family
from shm.globals import INTERVENTION, GENE, CONDITION
from shm.link import Link
from shm.model import Model
from shm.sampler import Sampler
from shm.step_methods.random_field_gibbs import RandomFieldGibbs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class SHM(ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 family=Family.gaussian,
                 link_function=Link.identity,
                 model=Model.simple,
                 n_states=2,
                 graph=None,
                 sampler=Sampler.metropolis):

        self._data = data
        self._graph = graph
        self._n_states = n_states
        if self._n_states not in [2, 3]:
            raise ValueError("Number of 'states' needs to be either 2 or 3")

        if graph:
            d_genes = sp.sort(sp.unique(self._data.gene.values))
            g_genes = sp.sort(graph.nodes())
            if not sp.array_equal(d_genes, g_genes):
                raise ValueError("Graph nodes != data genes")
            self._node_labels = d_genes

        self._set_data()
        self._set_link(link_function)
        self._set_sampler(sampler)
        self._set_family(family)
        self._set_model(model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def sample(self, draws=1000, tune=1000, chains=None, seed=23):
        with self.model:
            logger.info("Sampling {}/{} times".format(draws, tune))
            trace = pm.sample(
              draws=draws, tune=tune, chains=chains, cores=1,
              step=self._steps, random_seed=seed, progressbar=False)
        return trace

    @property
    def model(self):
        return self._model

    @property
    def steps(self):
        return self._steps

    @property
    def n_states(self):
        return self._n_states

    @property
    def model_type(self):
        return self._model_type

    @property
    def data(self):
        return self._data

    @property
    def graph(self):
        return self._graph

    @property
    def node_labels(self):
        return self._node_labels

    @property
    def family(self):
        return self._family

    @property
    def link(self):
        return self._link_function

    @property
    def sampler(self):
        return self._sampler

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

    @property
    def _beta_idx(self):
        return self.__beta_idx

    def _set_link(self, link_function):
        if isinstance(link_function, str):
            link_function = Link.from_str(link_function)
        self._link_function = link_function

    def _set_sampler(self, sampler):
        if isinstance(sampler, str):
            sampler = Sampler.from_str(sampler)
        self._sampler = sampler.value

    def _set_family(self, family):
        if isinstance(family, str):
            family = Family.from_str(family)
        self._family = family

    def _set_model(self, model):
        if isinstance(model, str):
            model = Model.from_str(model)
        self._model_type = model
        if model == Model.mrf:
            logger.info("Building mrf hierarchical model")
            if not self._graph:
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

    def _set_steps(self, model, z, *params):
        with model:
            self._continuous_step = self.sampler(params)
            if z is not None:
                if hasattr(z.distribution, "name") and \
                  z.distribution.name in [BinaryMRF.NAME, CategoricalMRF.NAME]:
                    self._discrete_step = RandomFieldGibbs([z])
                else:
                    self._discrete_step = pm.CategoricalGibbsMetropolis([z])
                self._steps = [self._continuous_step, self._discrete_step]
            else:
                self._steps = [self._continuous_step]
        self._model = model

    @abc.abstractmethod
    def _set_simple_model(self):
        pass

    @abc.abstractmethod
    def _set_clustering_model(self):
        pass

    @abc.abstractmethod
    def _set_mrf_model(self):
        pass

    @abc.abstractmethod
    def _hlm(self, model, gamma):
        pass

    def _gamma_mix(self, model, z):
        with model:
            tau_g = pm.InverseGamma(
              "tau_g", alpha=2., beta=1., shape=self.n_states)
            if self.n_states == 2:
                mean_g = pm.Normal(
                  "mu_g", mu=sp.array([-1., 0.]), sd=1, shape=2)
                pm.Potential(
                  "m_opot",
                  var=tt.switch(mean_g[1] - mean_g[0] < 0., -sp.inf, 0.))
            else:
                mean_g = pm.Normal(
                  "mu_g", mu=sp.array([-1, 0., 1.]), sd=1, shape=3)
                pm.Potential(
                  'm_opot',
                  tt.switch(mean_g[1] - mean_g[0] < 0, -sp.inf, 0)
                  + tt.switch(mean_g[2] - mean_g[1] < 0, -sp.inf, 0))

            gamma = pm.Normal("gamma", mean_g[z], tau_g[z], shape=self.n_genes)

        return tau_g, mean_g, gamma

    def _set_data(self):
        data = self._data
        self._n, _ = data.shape
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
