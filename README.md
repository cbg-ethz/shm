# Structured hierarchical models

> Deep hierarchical programs combined with Markov random fields.

## About

*Structured hierarchical models* (SHMs) utilize deep hierarchical models to represent heterogeneous data, and combine them with categorical Markov random fields (MRFs) to cluster the top-level latent variable of the hierarchy. The MRF to encode biological prior information over functionally related biological entities. The SHM is especially useful for extremely noisy, experimental data sets for which the true data generating process is difficult to model, for instance due to lacking domain knowledge or high stochasticity of the interventions.

An SHM has the following general form:

$$\begin{align}
\mathbf{z} & \sim \text{Categorical-MRF}\\
\boldsymbol \tau & \sim P_K(\tau) \\
\boldsymbol \mu & \sim \mathcal{N}_K(\mathbf{0}, \mathbf{I}) \\
\gamma_g & \mid z_g \sim \mathcal{N}(\mu_{z_g}, \tau_{z_g}^2) \\
x_{gci} & \mid \gamma_g \sim \text{HM}(\gamma_g)
\end{align}$$

The top-level categorical variables $z_g$ are latent cluster assignments for $g$ genes with effect sizes $\gamma_g$. The effect sizes are used to parameterize a deep hierarchical model, for instance to represent a heterogeneous data set of multiple conditions $c$ and interventions $i$, and readouts $x_{gci}$.

## Installation

You can install the package using:

```
pip install https://github.com/dirmeier/shm/archive/<version>.tar.gz
```

where `<version>` is the most recent release on GitHub. 
Please don't install from the master branch directly.

## Documentation

Check out the documentation [here](https://shm.readthedocs.io/en/latest/) to get you started with building your own SHM.

## Citation

If you like this package, please consider citing it:

> Dirmeier, S. and Beerenwinkel, N. (2019). Structured hierarchical models for probabilistic inference from noisy, heterogeneous data. *bioarvix*

## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier @ web.de</a>
