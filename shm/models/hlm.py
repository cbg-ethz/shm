import logging

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from pymc3.backends import NDArray
from pymc3.backends.base import MultiTrace

from shm.distributions.BinaryMRF import BinaryMRF
from shm.family import Family
from shm.globals import READOUT
from shm.link import Link
from shm.model import Model
from shm.models.hm import HM
from shm.step_methods.random_field_gibbs import RandomFieldGibbs

logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler())

class HLM(HM):
    def __init__(self,
                 data: pd.DataFrame,
                 family="gaussian",
                 link_function=Link.identity,
                 model="simple",
                 graph=None,
                 node_labels=None,
                 sampler="metropolis"):
        super().__init__(data=data,
                         family=family,
                         link_function=link_function,
                         model=model,
                         graph=graph,
                         node_labels=node_labels,
                         sampler=sampler)

    @property
    def model(self):
        return self.__model

    @property
    def _steps(self):
        return self.__steps

    def sample(self, n_draw=1000, n_tune=1000, seed=23):
        # TODO : add diagnostics
        # TODO: return a multitrace        
        with self.model:
            trace = pm.sample(50, tune=50, chains=1, cores=1,
            	          step=self._steps, random_seed=seed)
        return trace

    def _set_mrf_model(self):
        with pm.Model() as model:
            z = BinaryMRF('z', G=self.graph, node_labels=self.node_labels)

            tau_g = pm.InverseGamma("tau_g", alpha=5., beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=np.array([-1., 1.]), sd=0.5, shape=2)
            pm.Potential(
              "m_opot", var=tt.switch(mean_g[1] - mean_g[0] < 0, -np.inf, 0))
            gamma = pm.Normal("gamma", mean_g[z], tau_g, shape=self.n_genes)

            tau_b = pm.InverseGamma("tau_b", alpha=4., beta=1., shape=1)
            if self.n_conditions == 1:
                beta = pm.Deterministic("beta", var=gamma)
            else:
                beta = pm.Normal("beta",
                                 mu=gamma[self._beta_idx], sd=tau_b,
                                 shape=len(self._beta_idx))

            if self.family == Family.gaussian:
                l = pm.Normal("l", mu=0, sd=0.25, shape=self.n_interventions)
                sd = pm.HalfNormal("sd", sd=0.5)
                pm.Normal(
                  "x",
                  mu=beta[self._gene_cond_data_idx] + l[self._intervention_data_idx],
                  sd=sd,
                  observed=np.squeeze(self.data[READOUT].values))
            else:
                l = pm.Lognormal("l", mu=1, sd=0.25, shape=self.n_interventions)
                pm.Poisson(
                  "x",
                  mu=self.link(beta[self._gene_cond_data_idx]) *
                     l[self._intervention_data_idx],
                  observed=np.squeeze(self.data[READOUT].values))

        with model:
            self._discrete_step = RandomFieldGibbs([z])
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l, sd])
            else:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l])

        self.__steps = [self._discrete_step, self._continuous_step]
        self.__model = model
        return self

    def _set_clustering_model(self):
        with pm.Model() as model:
            p = pm.Dirichlet("p", a=np.array([1., 1.]), shape=2)
            pm.Potential("p_pot",
                         var=tt.switch(tt.min(p) < 0.05, -np.inf, 0))
            z = pm.Categorical("z", p=p, shape=self.n_genes)

            tau_g = pm.InverseGamma("tau_g", alpha=5., beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=np.array([-1., 1.]), sd=0.5, shape=2)
            pm.Potential(
              "m_opot", var=tt.switch(mean_g[1] - mean_g[0] < 0, -np.inf, 0))
            gamma = pm.Normal(
              "gamma", mean_g[z], tau_g, shape=self.n_genes)

            tau_b = pm.InverseGamma("tau_b", alpha=4., beta=1., shape=1)
            if self.n_conditions == 1:
                beta = pm.Deterministic("beta", var=gamma)
            else:
                beta = pm.Normal("beta",
                                 mu=gamma[self._beta_idx], sd=tau_b,
                                 shape=len(self._beta_idx))

            if self.family == Family.gaussian:
                l = pm.Normal("l", mu=0, sd=0.25, shape=self.n_interventions)
                sd = pm.HalfNormal("sd", sd=0.5)
                pm.Normal(
                  "x",
                  mu=beta[self._gene_cond_data_idx] + l[
                      self._intervention_data_idx],
                  sd=sd,
                  observed=np.squeeze(self.data[READOUT].values))
            else:
                l = pm.Lognormal("l", mu=1, sd=0.25, shape=self.n_interventions)
                pm.Poisson(
                  "x",
                  mu=self.link(beta[self._gene_cond_data_idx]) *
                     l[self._intervention_data_idx],
                  observed=np.squeeze(self.data[READOUT].values))

        with model:
            self._discrete_step = pm.CategoricalGibbsMetropolis([z])
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    p, tau_g, mean_g, gamma, tau_b, beta, l, sd])
            else:
                self._continuous_step = self.sampler([
                    p, tau_g, mean_g, gamma, tau_b, beta, l])

        self.__steps = [self._discrete_step, self._continuous_step]
        self.__model = model
        return self

    def _set_simple_model(self):
        with pm.Model() as model:
            tau_g = pm.InverseGamma("tau_g", alpha=5., beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=0, sd=0.5, shape=1)
            gamma = pm.Normal("gamma", mean_g, tau_g, shape=self.n_genes)

            if self.n_conditions == 1:
                beta = pm.Deterministic("beta", var=gamma)
            else:
                tau_b = pm.InverseGamma("tau_b", alpha=4., beta=1., shape=1)
                beta = pm.Normal("beta",
                                 mu=gamma[self._beta_idx], sd=tau_b,
                                 shape=len(self._beta_idx))

            if self.family == Family.gaussian:
                l = pm.Normal("l", mu=0, sd=0.25, shape=self.n_interventions)
                sd = pm.HalfNormal("sd", sd=0.5)
                pm.Normal(
                  "x",
                  mu=beta[self._gene_cond_data_idx] + l[self._intervention_data_idx],
                  sd=sd,
                  observed=np.squeeze(self.data[READOUT].values))
            else:
                l = pm.Lognormal("l", mu=1, sd=0.25, shape=self.n_interventions)
                pm.Poisson(
                  "x",
                  mu=self.link(beta[self._gene_cond_data_idx]) *
                     l[self._intervention_data_idx],
                  observed=np.squeeze(self.data[READOUT].values))

        with model:
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, beta, l, sd])
            else:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l])

        self.__steps = [self._continuous_step]
        self.__model = model
        return self
