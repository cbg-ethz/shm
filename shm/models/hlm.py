import logging

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

from shm.distributions.BinaryMRF import BinaryMRF
from shm.family import Family
from shm.globals import READOUT
from shm.link import Link
from shm.models.hm import HM
from shm.step_methods.random_field_gibbs import RandomFieldGibbs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class HLM(HM):
    def __init__(self,
                 data: pd.DataFrame,
                 family="gaussian",
                 link_function=Link.identity,
                 model="simple",
                 graph=None,
                 sampler="metropolis"):
        super().__init__(data=data,
                         family=family,
                         link_function=link_function,
                         model=model,
                         graph=graph,
                         sampler=sampler)

    @property
    def model(self):
        return self.__model

    @property
    def _steps(self):
        return self.__steps

    def sample(self, draws=1000, tune=1000, chains=None, seed=23):
        with self.model:
            logger.info("Sampling {}/{} times".format(draws, tune))
            trace = pm.sample(
              draws=draws, tune=tune, chains=chains, cores=1,
              step=self._steps, random_seed=seed, progressbar=False)
        return trace

    def __gamma_mix(self, model, z):
        with model:
            tau_g = pm.InverseGamma("tau_g", alpha=3., beta=1., shape=2)
            mean_g = pm.Normal("mu_g", mu=np.array([0., -1.]), sd=0.5, shape=2)
            pm.Potential(
              "m_opot", var=tt.switch(mean_g[1] - mean_g[0] > 0., -np.inf, 0.))
            gamma = pm.Normal("gamma", mean_g[z], tau_g[z], shape=self.n_genes)
        return tau_g, mean_g, gamma

    def __hlm(self, model, gamma):
        with model:
            tau_b = pm.InverseGamma("tau_b", alpha=3., beta=1., shape=1)
            beta = pm.Normal("beta", 0, sd=tau_b, shape=self.n_gene_condition)

            l_tau = pm.InverseGamma("tau_l", alpha=3., beta=1.)
            l = pm.Normal("l", mu=0, sd=l_tau, shape=self.n_interventions)
            mu = gamma[self._gene_data_idx] + \
                 beta[self._gene_cond_data_idx] + \
                 l[self._intervention_data_idx]

            if self.family == Family.gaussian:
                sd = pm.HalfNormal("sd", sd=0.5)
                pm.Normal("x", mu=mu, sd=sd,
                          observed=np.squeeze(self.data[READOUT].values))
            else:
                sd = None
                pm.Poisson("x", mu=self.link(mu),
                           observed=np.squeeze(self.data[READOUT].values))

        return tau_b, beta, l_tau, l, sd

    def _set_mrf_model(self):
        with pm.Model() as model:
            z = BinaryMRF('z', G=self.graph)
        tau_g, mean_g, gamma = self.__gamma_mix(model, z)
        tau_b, beta, l_tau, l, sd = self.__hlm(model, gamma)

        with model:
            self._discrete_step = RandomFieldGibbs([z])
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l_tau, l, sd])
            else:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l_tau, l])

        self.__steps = [self._discrete_step, self._continuous_step]
        self.__model = model
        return self

    def _set_clustering_model(self):
        with pm.Model() as model:
            p = pm.Dirichlet("p", a=np.array([1., 1.]), shape=2)
            pm.Potential("p_pot",
                         var=tt.switch(tt.min(p) < 0.05, -np.inf, 0.))
            z = pm.Categorical("z", p=p, shape=self.n_genes)
        tau_g, mean_g, gamma = self.__gamma_mix(model, z)
        tau_b, beta, l_tau, l, sd = self.__hlm(model, gamma)

        with model:
            self._discrete_step = pm.CategoricalGibbsMetropolis([z])
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    p, tau_g, mean_g, gamma, tau_b, beta, l_tau, l, sd])
            else:
                self._continuous_step = self.sampler([
                    p, tau_g, mean_g, gamma, tau_b, beta, l_tau, l])

        self.__steps = [self._discrete_step, self._continuous_step]
        self.__model = model
        return self

    def _set_simple_model(self):
        with pm.Model() as model:
            tau_g = pm.InverseGamma("tau_g", alpha=5., beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=0, sd=0.5, shape=1)
            gamma = pm.Normal("gamma", mean_g, tau_g, shape=self.n_genes)

        tau_b, beta, l_tau, l, sd = self.__hlm(model, gamma)
        with model:
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l_tau, l, sd])
            else:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l_tau, l])

        self.__steps = [self._continuous_step]
        self.__model = model
        return self
