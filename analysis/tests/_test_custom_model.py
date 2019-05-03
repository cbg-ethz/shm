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


pm.sample

class MyBinaryMRFSampler(ArrayStep):
    def __init__(self, vars, model=None):
        model = pm.modelcontext(model)
        vars = pm.inputvars(vars)
        super(MyBinaryMRFSampler, self).__init__(vars, [model.fastlogp])

    def step(self, point):
        print("Stepping")
        point["ni"] = np.array(self.vars[0].random(point))
        return point


with pm.Model() as m:
    ni = MyDistr('ni', .5)
    ps = pm.Uniform('ps', 0., .5, shape=1)
    p = pm.Uniform('p', ps, 1., shape=1)
    k = pm.Binomial('k', p=p, n=(ni + 1) * 10, observed=4)


print("now1")
with m:
    step1 = pm.NUTS([p, ps])
    step2 = MyBinaryMRFSampler([ni])

print("now2")
with m:
    pm.sample(50, tune=50, step=[step1, step2], chains=1, cores=1)

print("now3")
point = m.test_point
print(point)

for i in range(10):
    point = step2.step(point)
    point, _ = step1.step(point)
    print(point)
