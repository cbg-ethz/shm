import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from pymc3.backends import NDArray
from sklearn.preprocessing import LabelEncoder

from shm.family import Family
from shm.globals import READOUT, CONDITION, INTERVENTION
from shm.link import Link
from shm.models.hm import HM
from shm.distributions.BinaryMRF import BinaryMRF
from shm.sampler import Sampler
from shm.step_methods.random_field_gibbs import RandomFieldGibbs


class HLM(HM):
    def __init__(self,
                 data: pd.DataFrame,
                 independent_interventions=False,
                 family=Family.gaussian,
                 link=Link.identity,
                 graph=None,
                 node_labels=None,
                 sampler=Sampler.NUTS):
        super().__init__(data,
                         independent_interventions,
                         family,
                         link,
                         sampler)
        self.__graph = graph
        self.__node_labels = node_labels
        self._is_set = False

    def __enter__(self):
        if not self._is_set:
            self._set_model()
            self._set_samplers()
            self._is_set = True
        return self


    def sample(self, n_draw=1000, n_tune=1000, random_seed=23):
        # TODO : add diagnostics
        # TODO: return a multitrace
        trace = NDArray(model=self.__model)
        point = pm.Point(self.__model.test_point, model=self.__model)
        for i in range(n_tune + n_draw):
            point = self._mrf_step(point)
            point, state = self.__param_step(point)
            trace.record(point, state)

    def _set_model(self):
        if self.__graph:
            self._set_mrf_model()
        else:
            self._set_clustering_model()

    def _set_mrf_model(self):
        with pm.Model() as model:
            z = BinaryMRF('z', G=self.__graph, node_labels=self.__node_labels)

            tau_g = pm.InverseGamma("tau_g", alpha=5., beta=1., shape=1)
            mean_g = pm.Normal("mu_g", mu=np.array([-1., 1.]), sd=0.5, shape=2)
            pm.Potential(
              "m_opot", var=tt.switch(mean_g[1] - mean_g[0] < 0, -sp.inf, 0))
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
                  mu=self.link(beta[self.data[CONDITION]]) * l[self.data[INTERVENTION]],
                  observed=np.squeeze(self.data[READOUT].values))

        with model:
            self._mrf_step = RandomFieldGibbs([z])
            if self.family == Family.gaussian:
                self._param_step = self.sampler([tau_g, mean_g, gamma,
                                                 tau_b, beta, l, sd])
            else:
                self._param_step = self.sampler([tau_g, mean_g, gamma,
                                                 tau_b, beta, l])

        self.__model = model
        return self

    def _set_clustering_model(self):
        with pm.Model() as model:
            if self.__graph:
                z = BinaryMRF('z', G=self.__graph,
                              node_labels=self.__node_labels)
            else:
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
                beta = pm.Deterministic(
                  "beta",
                  var=gamma)
            else:
                beta = pm.Normal("beta",
                                 mu=gamma[self.beta_idx],
                                 sd=tau_b,
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
                  mu=self.link(beta[self.data[CONDITION]]) * l[
                      self.data[INTERVENTION]],
                  observed=np.squeeze(self.data[READOUT].values))

        with model:
            if self.__graph:

            step1 = MyBinaryMRFSampler([z])
            step2 = pm.Metropolis([p, ps])

        self.__model = model
        return self


def shm_clustering(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = np.unique(read_counts["Condition"].values)
    genes = np.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(np.unique(con_idx))
    len_sirnas = len(np.unique(read_counts["sgRNA"].values))
    len_replicates = len(np.unique(read_counts["replicate"].values))
    len_sirnas_per_gene = int(len_sirnas / len_genes)

    beta_idx = np.repeat(range(len_genes), len_conditions)
    beta_data_idx = np.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[np.repeat(np.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    l_idx = np.repeat(
      range(len_genes * len_conditions * len_sirnas_per_gene), len_replicates)

    with pm.Model() as model:
        p = pm.Dirichlet("p", a=np.array([1.0, 1.0]), shape=2)
        pm.Potential("p_pot", tt.switch(tt.min(p) < 0.05, -np.inf, 0))
        category = pm.Categorical("category", p=p, shape=len_genes)

        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        mean_g = pm.Normal("mu_g", mu=np.array([0, 0]), sd=0.5, shape=2)
        pm.Potential("m_opot", tt.switch(mean_g[1] - mean_g[0] < 0, -np.inf, 0))
        gamma = pm.Normal("gamma", mean_g[category], tau_g, shape=len_genes)

        tau_b = pm.InverseGamma("tau_b", 2.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b, shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=len_sirnas)
            sd = pm.HalfNormal("sd", sd=0.5)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l[l_idx],
              sd=sd,
              observed=np.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=len_sirnas)
            pm.Poisson(
              "x",
              mu=np.exp(beta[beta_data_idx]) * l[l_idx],
              observed=np.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds


def shm_clustering_independent_l(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = np.unique(read_counts["Condition"].values)
    genes = np.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))

    beta_idx = sp.repeat(range(len_genes), len_conditions)
    beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    with pm.Model() as model:
        p = pm.Dirichlet("p", a=sp.array([1.0, 1.0]), shape=2)
        pm.Potential("p_pot", tt.switch(tt.min(p) < 0.05, -sp.inf, 0))
        category = pm.Categorical("category", p=p, shape=len_genes)

        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        mean_g = pm.Normal("mu_g", mu=sp.array([0, 0]), sd=0.5, shape=2)
        pm.Potential("m_opot", tt.switch(mean_g[1] - mean_g[0] < 0, -sp.inf, 0))
        gamma = pm.Normal("gamma", mean_g[category], tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=n)
            sd = pm.HalfNormal("sd", sd=0.5)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l,
              sd=sd,
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=n)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l,
              observed=sp.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds


def shm_no_clustering(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = sp.unique(read_counts["Condition"].values)
    genes = sp.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))
    len_sirnas = len(sp.unique(read_counts["sgRNA"].values))
    len_replicates = len(sp.unique(read_counts["replicate"].values))
    len_sirnas_per_gene = int(len_sirnas / len_genes)

    beta_idx = sp.repeat(range(len_genes), len_conditions)
    beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    l_idx = sp.repeat(
      range(len_genes * len_conditions * len_sirnas_per_gene), len_replicates)

    with pm.Model() as model:
        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        gamma = pm.Normal("gamma", 0, tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=len_sirnas)
            sd = pm.HalfNormal("sd", sd=0.5)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l[l_idx],
              sd=sd,
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=len_sirnas)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l[l_idx],
              observed=sp.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds


def shm_no_clustering_independent_l(read_counts: pd.DataFrame, normalize):
    n, _ = read_counts.shape
    le = LabelEncoder()

    conditions = sp.unique(read_counts["Condition"].values)
    genes = sp.unique(read_counts["Gene"].values)
    gene_idx = le.fit_transform(read_counts["Gene"].values)
    con_idx = le.fit_transform(read_counts["Condition"].values)

    len_genes = len(sp.unique(gene_idx))
    len_conditions = len(sp.unique(con_idx))

    beta_idx = sp.repeat(range(len_genes), len_conditions)
    beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

    con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
    gene_conds = ["{}-{}".format(a, b) for a, b in zip(genes[beta_idx], con)]

    with pm.Model() as model:
        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        gamma = pm.Normal("gamma", 0, tau_g, shape=len_genes)

        tau_b = pm.Gamma("tau_b", 1.0, 1.0, shape=1)
        if len_conditions == 1:
            beta = pm.Deterministic("beta", gamma)
        else:
            beta = pm.Normal("beta", gamma[beta_idx], tau_b,
                             shape=len(beta_idx))

        if normalize:
            l = pm.Normal("l", 0, 0.25, shape=n)
            sd = pm.HalfNormal("sd", sd=0.5)
            pm.Normal(
              "x",
              mu= beta[beta_data_idx] + l,
              sd=sd,
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=n)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l,
              observed=sp.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds
