
import scipy.stats
import pymc3
from pymc3.step_methods.arraystep import ArrayStep


class RandomFieldGibbs(ArrayStep):
    name = 'random_field_gibbs'

    def __init__(self, var, model=None):
        self.__model = pymc3.modelcontext(model)
        vars = pymc3.inputvars(var)
        if len(var) != 1:
            raise ValueError("Please provide only one")

        self.__var = var[0]
        self.__var_name = self.__var.name

    def step(self, point):
        # TODO parserino of parameterino
        z = point[self.__var_name]
        point[self.__var_name] = self.__var.distribution.posterior_sample(point)
        return point

