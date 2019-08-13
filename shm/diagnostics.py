# __author__ = 'Simon Dirmeier'
# __email__  = 'simon.dirmeier@bsse.ethz.ch'
# __date__   = 03.07.19

import pandas as pd
import numpy as np
import arviz as az


def n_eff(trace, var_name):
    eff_samples = (az.ess(data=trace, var_names=var_name, relative=True)).to_dataframe()
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
