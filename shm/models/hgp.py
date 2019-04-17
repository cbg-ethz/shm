import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from pymc3.backends import NDArray

from shm.distributions.BinaryMRF import BinaryMRF
from shm.family import Family
from shm.globals import READOUT, CONDITION, INTERVENTION
from shm.link import Link
from shm.models.hm import HM
from shm.sampler import Sampler
from shm.step_methods.random_field_gibbs import RandomFieldGibbs


class HGP(HM):
    def __init__(self,
                 data: pd.DataFrame,
                 family=Family.gaussian,
                 link=Link.identity,
                 graph=None,
                 node_labels=None,
                 sampler=Sampler.NUTS):
        super().__init__(data,
                         family,
                         link,
                         graph,
                         node_labels,
                         sampler)

    def _set_mrf_model(self):
        with pm.Model() as model:
            z = BinaryMRF('z', G=self.__graph, node_labels=self.__node_labels)

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
                                 mu=gamma[self.beta_idx], sd=tau_b,
                                 shape=len(self._beta_idx))

            if self.family == Family.gaussian:
                l = pm.Normal("l", mu=0, sd=0.25, shape=self.n_interventions)
                sd = pm.HalfNormal("sd", sd=0.5)
                pm.Normal(
                  "x",
                  mu=beta[self.data[CONDITION]] + l[self.data[INTERVENTION]],
                  sd=sd,
                  observed=np.squeeze(self.data[READOUT].values))
            else:
                l = pm.Lognormal("l", mu=1, sd=0.25, shape=self.n_interventions)
                pm.Poisson(
                  "x",
                  mu=self.link(beta[self.data[CONDITION]]) *
                     l[self.data[INTERVENTION]],
                  observed=np.squeeze(self.data[READOUT].values))

        with model:
            self._discrete_step = RandomFieldGibbs([z])
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l, sd])
            else:
                self._continuous_step = self.sampler([
                    tau_g, mean_g, gamma, tau_b, beta, l])

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
                                 mu=gamma[self.beta_idx],
                                 sd=tau_b,
                                 shape=len(self.beta_idx))

            if self.family == Family.gaussian:
                l = pm.Normal("l", mu=0, sd=0.25, shape=self.n_interventions)
                sd = pm.HalfNormal("sd", sd=0.5)
                pm.Normal(
                  "x",
                  mu=beta[self.data[CONDITION]] + l[self.data[INTERVENTION]],
                  sd=sd,
                  observed=np.squeeze(self.data[READOUT].values))
            else:
                l = pm.Lognormal("l", mu=1, sd=0.25, shape=self.n_interventions)
                pm.Poisson(
                  "x",
                  mu=self.link(beta[self.data[CONDITION]]) * \
                     l[self.data[INTERVENTION]],
                  observed=np.squeeze(self.data[READOUT].values))

        with model:
            self._discrete_step = pm.CategoricalGibbsMetropolis([z])
            if self.family == Family.gaussian:
                self._continuous_step = self.sampler([
                    p, tau_g, mean_g, gamma, tau_b, beta, l, sd])
            else:
                self._continuous_step = self.sampler([
                    p, tau_g, mean_g, gamma, tau_b, beta, l])

        self.__model = model
        return self
