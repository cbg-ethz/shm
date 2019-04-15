

import pandas as pd
import pymc3 as pm
import scipy as sp
import theano.tensor as tt
from sklearn.preprocessing import LabelEncoder

from shm.family import Family
from shm.globals import READOUT, CONDITION, INTERVENTION
from shm.link import Link
from shm.models.hm import HM


class HLM(HM):
    def __init__(self,
                 data: pd.DataFrame,
                 independent_interventions=False,
                 family=Family.gaussian,
                 link=Link.identity,
                 graph=None,
                 node_labels=None):
        super().__init__(data,
                         independent_interventions,
                         family,
                         link,
                         graph,
                         node_labels)

    def __enter__(self):
        if self.do_random_field:
            self._set_mrf_hlm()
        else:
            self._set_mixture_hlm()
        return self

    # def shm(count_table, len_g, len_beta, len_l):
    #     p = ed.Uniform(0., 1., name="p")
    #     category = ed.Bernoulli(probs=tf.ones(len_g) * p, name="category")
    #
    #     gamma_tau = ed.InverseGamma(5, 1, name="gamma_tau")
    #     gamma = ed.Normal(tf.constant([-1., 1.]), gamma_tau, name="gamma")
    #
    #     beta_tau = ed.InverseGamma(5, 1, name="beta_tau")
    #     beta = ed.Normal(tf.gather(gamma, np.array([0, 0, 1, 1])),
    #                      beta_tau, name="beta")
    #     beta_f = tf.gather(beta, count_table['conditions'])
    #
    #     l = ed.Normal(tf.zeros(len_l), .1, name="l")
    #     le = tf.gather(l, count_table['sgrnas'])
    #
    #     x = ed.Poisson(le + tf.exp(beta_f), name="x")
    #     return x
    #
    # log_joint = ed.make_log_joint_fn(shm)
    #
    # def target_log_prob_fn(p, category, gamma_tau, gamma, beta_tau, beta, l):
    #     return log_joint(
    #       count_table=count_table,
    #       len_g=2, len_beta=4, len_l=8,
    #       p=p,
    #       category=category,
    #       gamma_tau=gamma_tau,
    #       gamma=gamma,
    #       beta_tau=beta_tau,
    #       beta=beta,
    #       l=l,
    #       x=v[0])
    #
    # tf.reset_default_graph()
    #
    # num_results = int(10e5)
    # num_burnin_steps = int(10e5)
    #
    # states, kernel_results = tfp.mcmc.sample_chain(
    #   num_results=num_results,
    #   num_burnin_steps=num_burnin_steps,
    #   current_state=[
    #       tf.ones([], name='p'),
    #       tf.ones(2, name='category'),
    #       tf.ones([], name='gamma_tau'),
    #       tf.zeros(2, name='gamma'),
    #       tf.ones([], name='beta_tau'),
    #       tf.zeros(4, name='beta'),
    #       tf.zeros(8, name='l')
    #   ],
    #   kernel=tfp.mcmc.RandomWalkMetropolis(
    #     target_log_prob_fn=target_log_prob_fn)
    # );
    #
    # p_, category_, gamma_tau_, gamma_, beta_tau_, beta_, l_ = states
    #
    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     init_op.run()
    #     [
    #         states, is_accepted_
    #     ] = sess.run([
    #         states, kernel_results.is_accepted
    #     ])
    #
    # def _set_mrf_hlm(self):

    def _set_mixture_hlm(self):
        with pm.Model() as model:
            p = pm.Dirichlet("p", a=sp.array([1., 1.]), shape=2)
            pm.Potential("p_pot", tt.switch(tt.min(p) < 0.05, -sp.inf, 0))
            category = pm.Categorical("category", p=p, shape=self.n_genes)

            tau_g = pm.InverseGamma("tau_g", 5., 1., shape=1)
            mean_g = pm.Normal("mu_g", mu=sp.array([-1., 1.]), sd=0.5, shape=2)
            pm.Potential(
              "m_opot", tt.switch(mean_g[1] - mean_g[0] < 0, -sp.inf, 0))
            gamma = pm.Normal(
              "gamma", mean_g[category], tau_g, shape=self.n_genes)

            tau_b = pm.InverseGamma("tau_b", 4., 1., shape=1)
            if self.n_conditions == 1:
                beta = pm.Deterministic(
                  "beta",
                  gamma)
            else:
                beta = pm.Normal(
                  "beta",
                  gamma[self.beta_idx],
                  tau_b,
                  shape=len(self._beta_idx))

            if self.family == Family.gaussian:
                l = pm.Normal("l", 0, 0.25, shape=self.n_interventions)
                sd = pm.HalfNormal("sd", sd=0.5)
                pm.Normal(
                  "x",
                  mu=beta[self.data[CONDITION]] + l[self.data[INTERVENTION]],
                  sd=sd,
                  observed=sp.squeeze(self.data[READOUT].values),
                )
            else:
                l = pm.Lognormal("l", 0, 0.25, shape=self.n_interventions)
                pm.Poisson(
                  "x",
                  mu=sp.exp(beta[self.data[CONDITION]]) * l[self.data[INTERVENTION]],
                  observed=sp.squeeze(read_counts["counts"].values))
        self.__model = model
        return self




def shm_clustering(read_counts: pd.DataFrame, normalize):
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
        p = pm.Dirichlet("p", a=sp.array([1.0, 1.0]), shape=2)
        pm.Potential("p_pot", tt.switch(tt.min(p) < 0.05, -sp.inf, 0))
        category = pm.Categorical("category", p=p, shape=len_genes)

        tau_g = pm.Gamma("tau_g", 1.0, 1.0, shape=1)
        mean_g = pm.Normal("mu_g", mu=sp.array([0, 0]), sd=0.5, shape=2)
        pm.Potential("m_opot", tt.switch(mean_g[1] - mean_g[0] < 0, -sp.inf, 0))
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
              observed=sp.squeeze(read_counts["counts"].values),
            )
        else:
            l = pm.Lognormal("l", 0, 0.25, shape=len_sirnas)
            pm.Poisson(
              "x",
              mu=sp.exp(beta[beta_data_idx]) * l[l_idx],
              observed=sp.squeeze(read_counts["counts"].values))

    return model, genes, gene_conds


def shm_clustering_independent_l(read_counts: pd.DataFrame, normalize):
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
