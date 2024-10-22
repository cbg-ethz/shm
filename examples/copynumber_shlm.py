# Copyright (C) 2018, 2019 Simon Dirmeier
#
# This file is part of shm.
#
# shm is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# shm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with shm. If not, see <http://www.gnu.org/licenses/>.
#
# @author = 'Simon Dirmeier'
# @email = 'simon.dirmeier@bsse.ethz.ch'

import logging

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

from shm.family import Family
from shm.globals import READOUT, AFFINITY, COPYNUMBER
from shm.link import Link
from examples.shlm import SHLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class CopynumberSHLM(SHLM):
    def __init__(self,
                 data: pd.DataFrame,
                 family="gaussian",
                 link_function=Link.identity,
                 model="clustering",
                 n_states=2,
                 graph=None,
                 sampler="nuts",
                 affinity="data",
                 edge_cor=.15):
        self._affinity = affinity
        self.__edge_cor = edge_cor
        super().__init__(data=data,
                         family=family,
                         link_function=link_function,
                         model=model,
                         n_states=n_states,
                         graph=graph,
                         sampler=sampler)

    @property
    def tau_g_alpha(self):
        return 5

    @property
    def tau_b_alpha(self):
        return 3

    @property
    def tau_iota_alpha(self):
        return 3

    @property
    def sd_alpha(self):
        return 2

    @property
    def kappa_sd(self):
        return 1

    @property
    def edge_correction(self):
        return self.__edge_cor

    @property
    def gamma_means(self):
        if self.n_states == 2:
            return np.array([0., 0.])
        return np.array([-1, 0., 1.])

    def _gamma_mix(self, model, z):
        with model:
            logger.info("Using tau_g_alpha: {}".format(self.tau_g_alpha))
            tau_g = pm.InverseGamma(
              "tau_g", alpha=self.tau_g_alpha, beta=1., shape=self.n_states)

            logger.info("Using mean_g: {}".format(self.gamma_means))
            if self.n_states == 2:
                logger.info("Building two-state model")
                mean_g = pm.Normal(
                  "mu_g", mu=self.gamma_means, sd=1, shape=self.n_states)
                pm.Potential(
                  "m_opot",
                  var=tt.switch(mean_g[1] - mean_g[0] < 0., -np.inf, 0.))
            else:
                logger.info("Building three-state model")
                mean_g = pm.Normal(
                  "mu_g", mu=self.gamma_means, sd=1, shape=self.n_states)
                pm.Potential(
                  'm_opot',
                  tt.switch(mean_g[1] - mean_g[0] < 0, -np.inf, 0)
                  + tt.switch(mean_g[2] - mean_g[1] < 0, -np.inf, 0))

            gamma = pm.Normal("gamma", mean_g[z], tau_g[z], shape=self.n_genes)

        return tau_g, mean_g, gamma

    def _hlm(self, model, gamma):
        with model:
            logger.info("Using tau_b_alpha: {}".format(self.tau_b_alpha))
            tau_b = pm.InverseGamma(
              "tau_b", alpha=self.tau_b_alpha, beta=1., shape=1)
            beta = pm.Normal("beta", 0, sd=tau_b, shape=self.n_gene_condition)

            logger.info("Using tau_iota_alpha: {}".format(self.tau_iota_alpha))
            l_tau = pm.InverseGamma(
              "tau_iota", alpha=self.tau_iota_alpha, beta=1., shape=1)
            l = pm.Normal("iota", mu=0, sd=l_tau, shape=self.n_interventions)

            logger.info("Using kappa_sd: {}".format(self.kappa_sd))
            c = pm.Normal("kappa", 0, self.kappa_sd, shape=1)

            if self._affinity == "data":
                logger.info("Using affinity from data")
                q = self.data[AFFINITY].values
            elif self._affinity == "leaveout":
                logger.info("Using no affinity")
                q = 1
            elif self._affinity == "estimate":
                logger.info("Estimating affinity from data")
                q = pm.Uniform(
                  "aff", lower=0, upper=1, shape=self.n_interventions)
            else:
                raise ValueError("Wrong affinity")

            mu = l[self._intervention_data_idx]

            ll = (gamma[self._gene_data_idx] +
                  beta[self._gene_cond_data_idx] +
                  c * self.data[COPYNUMBER].values)
            if self._affinity == "estimate":
                mu += q[self._intervention_data_idx] * ll
            else:
                mu += q * ll

            if self.family == Family.gaussian:
                logger.info("Using sd_alpha: {}".format(self.sd_alpha))
                sd = pm.InverseGamma("sd", alpha=self.sd_alpha, beta=1.,
                                     shape=1)
                pm.Normal("x",
                          mu=mu,
                          sd=sd,
                          observed=np.squeeze(self.data[READOUT].values))
            else:
                raise NotImplementedError("Only gaussian family so far")

        if self._affinity == "estimate":
            return tau_b, beta, l_tau, l, sd, q, c
        return tau_b, beta, l_tau, l, sd, c
