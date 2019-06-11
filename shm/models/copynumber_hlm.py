import logging

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

from shm.distributions.binary_mrf import BinaryMRF
from shm.family import Family
from shm.globals import READOUT, AFFINITY, COPYNUMBER
from shm.link import Link
from shm.models.hlm import HLM
from shm.step_methods.random_field_gibbs import RandomFieldGibbs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class CopynumberHLM(HLM):
    def __init__(self,
                 data: pd.DataFrame,
                 family="gaussian",
                 link_function=Link.identity,
                 model="simple",
                 n_states=2,
                 graph=None,
                 sampler="metropolis",
                 use_affinity=False):
        self._use_affinity = use_affinity
        super().__init__(data=data,
                         family=family,
                         link_function=link_function,
                         model=model,
                         n_states=n_states,
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
            tau_g = pm.InverseGamma("tau_g", alpha=2., beta=1., shape=2)
            mean_g = pm.Normal("mu_g", mu=np.array([0., -1.]), sd=1, shape=2)
            pm.Potential(
              "m_opot", var=tt.switch(mean_g[1] - mean_g[0] > 0., -np.inf, 0.))
            gamma = pm.Normal("gamma", mean_g[z], tau_g[z], shape=self.n_genes)

        return tau_g, mean_g, gamma

    def __hlm(self, model, gamma):
        with model:
            tau_b = pm.InverseGamma("tau_b", alpha=2., beta=1., shape=1)
            beta = pm.Normal("beta", 0, sd=tau_b, shape=self.n_gene_condition)

            l_tau = pm.InverseGamma("tau_l", alpha=2., beta=1., shape=1)
            l = pm.Normal("l", mu=0, sd=l_tau, shape=self.n_interventions)
            c = pm.Normal("cn", 1, 1, shape=1)

            if self._use_affinity:
                q = self.data[AFFINITY].values
            else:
                q = 1
            mu = q * (gamma[self._gene_data_idx] +
                      beta[self._gene_cond_data_idx]) + \
                 l[self._intervention_data_idx] + \
                 c * self.data[COPYNUMBER].values

            if self.family == Family.gaussian:
                sd = pm.InverseGamma("sd", alpha=2., beta=1., shape=1)
                pm.Normal("x",
                          mu=mu,
                          sd=sd,
                          observed=np.squeeze(self.data[READOUT].values))
            else:
                raise NotImplementedError("Only gaussian family so far")

        return tau_b, beta, l_tau, l, sd, c

    def _set_mrf_model(self):
        with pm.Model() as model:
            z = BinaryMRF('z', G=self.graph)
        tau_g, mean_g, gamma = self.__gamma_mix(model, z)
        param_hlm = self.__hlm(model, gamma)
        self._set_steps(model, z, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _set_clustering_model(self):
        with pm.Model() as model:
            p = pm.Dirichlet("p", a=np.array([1., 1.]), shape=2)
            pm.Potential("p_pot", var=tt.switch(tt.min(p) < 0.05, -np.inf, 0.))
            z = pm.Categorical("z", p=p, shape=self.n_genes)
        tau_g, mean_g, gamma = self.__gamma_mix(model, z)
        param_hlm = self.__hlm(model, gamma)
        self._set_steps(model, z, p, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _set_steps(self, model, z, *params):
        with model:
            self._continuous_step = self.sampler(params)
            if z is not None:
                if hasattr(z.distribution, "name") and \
                  z.distribution.name == BinaryMRF.NAME:
                    self._discrete_step = RandomFieldGibbs([z])
                else:
                    self._discrete_step = pm.CategoricalGibbsMetropolis([z])
                self.__steps = [self._continuous_step, self._discrete_step]
            else:
                self.__steps = [self._continuous_step]
        self.__model = model

    def _set_simple_model(self):
        with pm.Model() as model:
            tau_g = pm.InverseGamma("tau_g", alpha=2., beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=0, sd=1, shape=1)
            gamma = pm.Normal("gamma", mean_g, tau_g, shape=self.n_genes)
        param_hlm = self.__hlm(model, gamma)
        self._set_steps(model, None, tau_g, mean_g, gamma, *param_hlm)
        return self
