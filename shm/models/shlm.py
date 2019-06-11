import logging

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

from shm.distributions.binary_mrf import BinaryMRF
from shm.distributions.categorical_mrf import CategoricalMRF
from shm.family import Family
from shm.globals import READOUT
from shm.link import Link
from shm.models.shm import SHM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class SHLM(SHM):
    def __init__(self,
                 data: pd.DataFrame,
                 family="gaussian",
                 link_function=Link.identity,
                 model="simple",
                 n_states=2,
                 graph=None,
                 sampler="metropolis"):
        super().__init__(data=data,
                         family=family,
                         link_function=link_function,
                         model=model,
                         n_states=n_states,
                         graph=graph,
                         sampler=sampler)

    def _set_mrf_model(self):
        with pm.Model() as model:
            if self.n_states == 2:
                z = BinaryMRF('z', G=self.graph)
            else:
                z = CategoricalMRF('z', G=self.graph, k=3)
        tau_g, mean_g, gamma = self._gamma_mix(model, z)
        param_hlm = self._hlm(model, gamma)

        self._set_steps(model, z, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _set_clustering_model(self):
        with pm.Model() as model:
            p = pm.Dirichlet("p", a=np.array([1., 1.]), shape=2)
            pm.Potential("p_pot", var=tt.switch(tt.min(p) < 0.05, -np.inf, 0.))
            z = pm.Categorical("z", p=p, shape=self.n_genes)
        tau_g, mean_g, gamma = self._gamma_mix(model, z)
        param_hlm = self._hlm(model, gamma)

        self._set_steps(model, z, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _set_simple_model(self):
        with pm.Model() as model:
            tau_g = pm.InverseGamma("tau_g", alpha=2., beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=0, sd=1, shape=1)
            gamma = pm.Normal("gamma", mean_g, tau_g, shape=self.n_genes)
        param_hlm = self.__hlm(model, gamma)

        self._set_steps(model, None, tau_g, mean_g, gamma, *param_hlm)
        return self

    def _hlm(self, model, gamma):
        with model:
            tau_b = pm.InverseGamma("tau_b", alpha=2., beta=1., shape=1)
            beta = pm.Normal("beta", 0, sd=tau_b, shape=self.n_gene_condition)

            l_tau = pm.InverseGamma("tau_l", alpha=2., beta=1., shape=1)
            l = pm.Normal("l", mu=0, sd=l_tau, shape=self.n_interventions)

            mu = (gamma[self._gene_data_idx] + beta[self._gene_cond_data_idx]) + \
                 l[self._intervention_data_idx]

            if self.family == Family.gaussian:
                sd = pm.InverseGamma("sd", alpha=2., beta=1., shape=1)
                pm.Normal("x",
                          mu=mu,
                          sd=sd,
                          observed=np.squeeze(self.data[READOUT].values))
            else:
                raise NotImplementedError("Only gaussian family so far")

        return tau_b, beta, l_tau, l, sd
