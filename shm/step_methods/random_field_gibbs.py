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

import numpy as np
import pymc3
from pymc3.step_methods.arraystep import ArrayStep


class RandomFieldGibbs(ArrayStep):
    name = 'random_field_gibbs'

    def __init__(self, vars, model=None):
        model = pymc3.modelcontext(model)
        if len(vars) != 1:
            raise ValueError("Please provide only one")

        vars = pymc3.inputvars(vars)
        self.__var = vars[0]
        self.__var_name = self.__var.name
        super(RandomFieldGibbs, self).__init__(vars, [model.fastlogp])

    def step(self, point):
        z = point['z']
        mu_g = point['mu_g']
        tau_g = np.exp(point['tau_g_log__'])
        gamma = point['gamma']
        point['z'] = \
            self.__var.distribution.posterior_sample(z, gamma, mu_g, tau_g)
        return point
