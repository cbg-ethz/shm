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


import pandas as pd
import numpy as np


def compute_posterior_probabilities(trace, nstates=2):
    if nstates not in [2, 3]:
        raise ValueError("nstates needs to be in {2, 3}")
    states = ['0', '1', '2'] if nstates == 3 else ['0', '1']
    states = np.array(states)

    def f(x):
        s = pd.Series(x).value_counts()
        s /= np.sum(s)
        keys = list(map(str, s.keys()))
        d = {e: i for e, i in zip(keys, s)}
        for k in np.setdiff1d(states, keys):
            d[k] = 0.0
        return np.array([d[k] for k in sorted(d.keys())])

    probs = np.apply_along_axis(lambda x: f(x), 0, trace['z']).T
    return probs
