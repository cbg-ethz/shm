# Structured hierarchical models <img src="https://github.com/cbg-ethz/shm/blob/master/_fig/sticker_shm.png" align="right" width="160px"/>

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

> Deep hierarchical programs combined with categorical Markov random fields.

## About

*Structured hierarchical models* are a family of generative models for probabilistic inference of causal effects from genetic perturbation screens. SHMs utilize classical hierarchical models to represent heterogeneous data and combine them with categorical Markov random fields to encode biological prior information over functionally related genes. The random field induces a clustering of the genes which helps to inform inference of parameters in the hierarchical model. SHMs are designed for extremely noisy, experimental data sets for which the true data generating process is difficult to model.

An SHM has the following basic structure.

<div align="center">
	<img src="https://github.com/cbg-ethz/shm/blob/master/_fig/model.png" width="300px"/>
</div>

The top-level categorical variables $z_g$ are latent cluster assignments for $g$ genes with effect sizes $\gamma_g$. The effect sizes are used to parameterize a deep hierarchical model, for instance to represent a heterogeneous data set $y_{gci}$ of multiple conditions $c$ and interventions $i$.

## Installation

You can install the package using:

```
pip install https://github.com/cbg-ethz/shm/archive/<version>.tar.gz
```

where `<version>` is the most recent release on GitHub. 
Please don't install from the master branch directly.


## Citation

If you like this package, please consider citing it:

> Dirmeier, S. and Beerenwinkel, N. (2019). Structured hierarchical models for probabilistic inference from perturbation screening data. *bioRxiv*. [https://doi.org/10.1101/848234](https://doi.org/10.1101/848234)

## Author

Simon Dirmeier <a href="mailto:simon.dirmeier@web.de">simon.dirmeier @ web.de</a>
