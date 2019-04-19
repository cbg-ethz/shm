from enum import Enum
import pymc3 as pm



import theano
import pymc3 as pm
import numpy as np
import scipy.stats
from pymc3 import Discrete
from pymc3.step_methods.arraystep import ArrayStep


class MyDistr(Discrete):
    def __init__(self, p=None, *args, **kwargs):
        super(MyDistr, self).__init__(*args, **kwargs)
        self.mode = 1

    def random(self, point=None, size=None):
        print("Sampling")
        return scipy.stats.bernoulli.rvs(.5)

    def logp(self, value):
        print("Only once")
        return 0


class MyBinaryMRFSampler(ArrayStep):
    def __init__(self, var):
        self.var = var[0]
        print(var)

    def step(self, p):
        print("Stepping")
        point["ni"] = np.array(self.var.random(point))
        return point


with pm.Model() as m:
    ni = MyDistr('ni', .5)
    s = pm.Poisson
    print("now5")
    ps = pm.Uniform('ps', 0., .5, shape=2)
    p = pm.Uniform('p', ps, 1., shape=2)
    print("now6")
    k = pm.Binomial('k', p=p[0], n=(ni + 1) * 10, observed=[4, 4])
    print("no7")


print("now1")
with m:
    step1 = pm.CategoricalGibbsMetropolis([p, ps])
    step2 = MyBinaryMRFSampler([ni])

pm.sample

print("now3")
point = m.test_point
print(point)

for i in range(10):
    point = step2.step(point)
    point, _ = step1.step(point)
    print(point)
