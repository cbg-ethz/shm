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
