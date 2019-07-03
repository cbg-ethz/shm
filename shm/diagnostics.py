# __author__ = 'Simon Dirmeier'
# __email__  = 'simon.dirmeier@bsse.ethz.ch'
# __date__   = 03.07.19

import pandas as pd
import arviz as az

def n_eff(trace, var_name):
    eff_samples = (az.effective_sample_size(trace)[var_name]).to_dataframe()
    boundary = [0.1, 0.5, 1]
    eff_samples = pd.DataFrame({
        "neff": eff_samples[var_name].values / (len(trace) * 4),
        "param": [var_name + str(i) for i in range(len(eff_samples))]})