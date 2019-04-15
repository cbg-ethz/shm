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
        return scipy.stats.bernoulli.rvs(p)

    def logp(self, value):
        print("Only once")
        return 0


class MyBinaryMRFSampler(ArrayStep):
    def __init__(self, vars):
        pass

    def step(self, p):
        return scipy.stats.bernoulli.rvs(p)

print("now4")
n_ = theano.shared(np.asarray([10, 15]))
with pm.Model() as m:
    ni = MyDistr('ni', .5)
    print("now5")
    p = pm.Uniform('p',0, 1.)
    print("now6")
    k = pm.Binomial('k', p=p, n=(ni + 1) * 10, observed=4)
    print("no7")

print("now1")
with m:
    step1 = pm.Metropolis([p])
    step2 = MyBinaryMRFSampler([ni])

print("now3")
point = m.test_point
print(point)

print("now2")

for i in range(10):
    point['ni'] = np.array(step2.step(.5))
    point, _ = step1.step(point)
    print(point)


