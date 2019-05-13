---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.1
  kernelspec:
    display_name: Python (ml)
    language: python
    name: ml
---

# Trace

Das klappt.

```python
import shm
import numpy
import networkx
import matplotlib
import seaborn as sns
import pandas as pd
import pickle
import pymc3 as pm
```

```python
import arviz as az
import shm.plot as sp
import matplotlib.pyplot as plt
import seaborn as sns
from shm.models.hlm import HLM
```

```python
from pymc3 import model_to_graphviz
```

```python
%pylab inline
pylab.rcParams['figure.figsize'] = (12, 3)
```

```python
sns.set_style(
  "white",
  {
      "xtick.bottom": True,
      "ytick.left": True,
      "axes.spines.top": False,
      "axes.spines.right": False,
  },
)
```

```python
def read_graph(infile):
    with open(infile, "rb") as fh:
        G = pickle.load(fh)
    return G
```

# Model evaluation

```python
readout_file = "../data_raw/simulated_data.tsv"
graph_file = "../data_raw/graph.pickle"
data_file = "../data_raw/data.pickle"
```

```python
with open(data_file, "rb") as fh:    
    data = pickle.load(fh)
```

```python
readout = pd.read_csv(readout_file, sep="\t")
G = read_graph(graph_file)
```

```python
networkx.draw(G, node_color="black")
```

```python
networkx.draw(networkx.subgraph(G, data['essential_genes']), node_color="black")
```

## Simple model

```python
trace_dir = "../../../results/simple_model_trace"
with HLM(readout, model="simple") as model:
    trace = pm.load_trace(trace_dir, model = model.model)
    ppc_trace = pm.sample_posterior_predictive(trace, 10000, model.model)
```

```python
data['gamma']
```

```python
sp.plot_hist(trace, "gamma", 0, "");
```

```python
sp.plot_hist(trace, "gamma", 15, "");
```

```python
trace
```

```python
data['gamma']
```

```python
numpy.mean(trace['gamma'], 0)[list(model._index_to_gene.keys())]
```

```python
sp.plot_steps(readout, ppc_trace, bins=19);
```

## Clustering model

```python
trace_dir = "../../../results/clustering_model_trace"
with HLM(readout, model="clustering") as model:
    trace = pm.load_trace(trace_dir, model = model.model)
    ppc_trace = pm.sample_posterior_predictive(trace, 10000, model.model)
```

```python
sp.plot_hist(trace, "gamma", 0, "");
```

```python
sp.plot_hist(trace, "gamma", 15, "");
```

```python
sp.plot_steps(readout, ppc_trace, bins=30);
```

```python
numpy.mean(trace['gamma'], 0)[list(model._index_to_gene.keys())]
```

```python
data['gamma']
```

```python
data['essential_genes']
```

```python
sp.plot_posterior_labels(
    trace, 
    [model._index_to_gene[x] for x in sorted(model._index_to_gene.keys())]);
```

```python
np.mean((readout["readout"].values - np.mean(ppc_trace['x'], 0))**2)
```

## MRF model

```python
trace_dir = "../../../results/mrf_model_trace"
with HLM(readout, model="mrf", graph=G) as model:
    trace = pm.load_trace(trace_dir, model = model.model)
    ppc_trace = pm.sample_posterior_predictive(trace, 10000, model.model)
```

```python
data['gamma']
```

```python
numpy.mean(trace['gamma'], 0)
```

```python
sp.plot_hist(trace, "gamma", 0, "");
```

```python
sp.plot_hist(trace, "gamma", 15, "");
```

```python
sp.plot_steps(readout, ppc_trace, bins=30);
```

```python
data['gamma']
```

```python
numpy.mean(trace['gamma'], 0)[list(model._index_to_gene.keys())]
```

```python
[model._index_to_gene[x] for x in sorted(model._index_to_gene.keys())]
```

```python
sp.plot_posterior_labels(trace, 
                         [model._index_to_gene[x] for x in sorted(model._index_to_gene.keys())]);
```

```python
sp.plot_neff(trace, "beta");
```

```python
sp.plot_rhat(trace, "gamma");
```

```python
np.mean((readout["readout"].values - np.mean(ppc_trace['x'], 0))**2)
```

```python

```
