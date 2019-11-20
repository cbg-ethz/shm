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
import arviz as az


def n_eff(trace, var_name):
    eff_samples = (az.ess(data=trace, var_names=var_name, relative=True)) \
        .to_dataframe()
    eff_samples = pd.DataFrame({
        "neff": eff_samples[var_name].values,
        "param": [var_name + str(i) for i in range(len(eff_samples))]})
    return eff_samples


def rhat(trace, var_name):
    rhat_samples = (az.rhat(data=trace, var_names=var_name)).to_dataframe()
    rhat_samples = pd.DataFrame({
        "rhat": rhat_samples[var_name].values,
        "param": [var_name + str(i) for i in range(len(rhat_samples))]})
    return rhat_samples


def cut_rhat(trace, var_name):
    rhat_samples = rhat(trace, var_name)
    rhat_samples = rhat_samples["rhat"].value_counts(
      bins=[-np.inf, 1.05, 1.1, np.inf], sort=False)
    rhat_samples = pd.DataFrame({
        "cut": [r"$[1, 1.05]$", r"$(1.05, 1.1]$", r"$(1.1, \infty)$"],
        "bins": rhat_samples.values
    })
    return rhat_samples


def cut_neff(trace, var_name):
    eff_samples = n_eff(trace, var_name)
    eff_samples = eff_samples["neff"].value_counts(
      bins=[-np.inf, .1, .5, 1, np.inf], sort=False)
    eff_samples = pd.DataFrame({
        "cut": [r"$[0, 0.1]$", r"$(0.1, 0.5]$", r"$(0.5, 1]$", r"$(0.5, 1]$"],
        "bins": eff_samples.values
    })
    return eff_samples
