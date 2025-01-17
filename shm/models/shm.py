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

import abc
import logging
from abc import ABC

import pymc3 as pm
import scipy

from shm.distributions.binary_mrf import BinaryMRF
from shm.distributions.categorical_mrf import CategoricalMRF
from shm.model import Model
from shm.sampler import Sampler
from shm.step_methods.random_field_gibbs import RandomFieldGibbs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class SHM(ABC):
    def __init__(self,
                 model=Model.clustering,
                 n_states=2,
                 graph=None,
                 sampler=Sampler.nuts):

        self._graph = graph
        self._node_labels = scipy.sort(graph.nodes())
        self._n_states = n_states
        if self._n_states not in [2, 3]:
            raise ValueError("Number of 'states' needs to be either 2 or 3")

        self._set_sampler(sampler)
        self._set_model(model)

    def __enter__(self):
        # TODO implement context manager to be more pymc like
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # nth do be done here i think
        pass

    def sample(self, draws=1000, tune=1000, chains=None, seed=23):
        with self.model:
            logger.info("Sampling tune:{}/draw:{} times".format(tune, draws))
            trace = pm.sample(
              draws=draws, tune=tune, chains=chains, cores=1,
              step=self._steps, random_seed=seed, progressbar=False)
        return trace

    @property
    def model(self):
        return self._model

    @property
    def steps(self):
        return self._steps

    @property
    def n_states(self):
        return self._n_states

    @property
    def model_type(self):
        return self._model_type

    @property
    def graph(self):
        return self._graph

    @property
    def node_labels(self):
        return self._node_labels

    @property
    def sampler(self):
        return self._sampler

    def _set_sampler(self, sampler):
        if isinstance(sampler, str):
            sampler = Sampler.from_str(sampler)
        self._sampler = sampler.value

    def _set_model(self, model):
        if isinstance(model, str):
            model = Model.from_str(model)
        self._model_type = model
        if model == Model.mrf:
            logger.info("Building mrf hierarchical model")
            if not self._graph:
                raise ValueError("You need to provide a graph")
            return self._set_mrf_model()
        elif model == Model.clustering:
            logger.info("Building cluster hierarchical model")
            return self._set_clustering_model()
        elif model == Model.simple:
            logger.info("Building simple hierarchical model")
            return self._set_simple_model()
        raise ValueError("Model not supported")

    def _set_steps(self, model, z, *params):
        with model:
            self._continuous_step = self.sampler(params)
            if z is not None:
                if hasattr(z.distribution, "name") and \
                  z.distribution.name in [BinaryMRF.NAME, CategoricalMRF.NAME]:
                    self._discrete_step = RandomFieldGibbs([z])
                else:
                    self._discrete_step = pm.CategoricalGibbsMetropolis([z])
                self._steps = [self._continuous_step, self._discrete_step]
            else:
                self._steps = [self._continuous_step]
        self._model = model

    @abc.abstractmethod
    def _set_clustering_model(self):
        pass

    @abc.abstractmethod
    def _set_mrf_model(self):
        pass

    @abc.abstractmethod
    def _hlm(self, model, gamma):
        pass

    @abc.abstractmethod
    def _gamma_mix(self, model, z):
        pass
