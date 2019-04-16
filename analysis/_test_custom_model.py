# import theano
# import pymc3 as pm
# import numpy as np
# import scipy.stats
# from pymc3 import Discrete
# from pymc3.step_methods.arraystep import ArrayStep
#
#
# class MyDistr(Discrete):
#     def __init__(self, p=None, *args, **kwargs):
#         super(MyDistr, self).__init__(*args, **kwargs)
#         self.mode = 1
#
#     def random(self, point=None, size=None):
#         print("Sampling")
#         return scipy.stats.bernoulli.rvs(.5)
#
#     def logp(self, value):
#         print("Only once")
#         return 0
#
#
# class MyBinaryMRFSampler(ArrayStep):
#     def __init__(self, var):
#         self.var = var[0]
#         print(var)
#
#     def step(self, p):
#         print("Stepping")
#         point["ni"] = np.array(self.var.random(point))
#         return point
#
#
# with pm.Model() as m:
#     ni = MyDistr('ni', .5)
#     print("now5")
#     ps = pm.Uniform('ps', 0., .5)
#     p = pm.Uniform('p', ps, 1.)
#     print("now6")
#
# with m:
#     k = pm.Binomial('k', p=p, n=(ni + 1) * 10, observed=4)
#     print("no7")
#
#
# print("now1")
# with m:
#     step1 = pm.NUTS([p, ps])
#     step2 = MyBinaryMRFSampler([ni])
#
# print("now3")
# point = m.test_point
# print(point)
#
# with m:
#     s = pm.sample(nchains=1, cores=1)
#
# print(type(s))
# print("now2")
#
# for i in range(10):
#     point = step2.step(point)
#     point, _ = step1.step(point)
#     print(point)
#
#

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import pymc3 as pm, theano.tensor as tt

np.random.seed(12345) # set random seed for reproducibility

k = 3
ndata = 500
spread = 5
centers = np.array([-spread, 0, spread])

# simulate data from mixture distribution
v = np.random.randint(0, k, ndata)
data = centers[v] + np.random.randn(ndata)

model = pm.Model()
with model:
    # cluster sizes
    p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=k)
    # ensure all clusters have some points
    p_min_potential = pm.Potential('p_min_potential',
                                   tt.switch(tt.min(p) < .1, -np.inf, 0))


    # cluster centers
    means = pm.Normal('means', mu=[0, 0, 0], sd=15, shape=k)
    # break symmetry
    order_means_potential = pm.Potential('order_means_potential',
                                         tt.switch(means[1]-means[0] < 0, -np.inf, 0)
                                         + tt.switch(means[2]-means[1] < 0, -np.inf, 0))

    # measurement error
    sd = pm.Uniform('sd', lower=0, upper=20)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=ndata)

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sd=sd,
                       observed=data)

with model:
    step1 = pm.CategoricalGibbsMetropolis([category])
    step2 = pm.Metropolis([p, means, sd])