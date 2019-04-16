import numpy
from abc import ABC

import pandas as pd
import pymc3
import scipy as sp
from sklearn.preprocessing import LabelEncoder

from shm.family import Family
from shm.globals import INTERVENTION, REPLICATE, GENE, CONDITION
from shm.link import Link
from shm.sampler import Sampler


class HM(ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 independent_interventions=False,
                 family=Family.gaussian,
                 link=Link.identity,
                 sampler=Sampler.NUTS):

        self.__data = data
        self.__link = link
        self.__family = family
        self.__independent_interventions = independent_interventions

        self.__n, _ = data.shape
        le = LabelEncoder()

        self.__conditions = sp.unique(data[CONDITION].values)
        self.__con_idx = le.fit_transform(data[CONDITION].values)

        self.__genes = sp.unique(data[GENE].values)
        self.__gene_idx = le.fit_transform(data[GENE].values)

        self.__len_genes = len(sp.unique(gene_idx))
        self.__len_conditions = len(sp.unique(con_idx))
        self.__len_sirnas = len(sp.unique(data[INTERVENTION].values))
        self.__len_replicates = len(sp.unique(data[REPLICATE].values))

        self.__beta_idx = sp.repeat(range(len_genes), len_conditions)
        self.__beta_data_idx = sp.repeat(beta_idx, int(n / len(beta_idx)))

        self.__con = conditions[sp.repeat(sp.unique(con_idx), len_genes)]
        self.__gene_conds = ["{}-{}".format(a, b) for a, b in
                             zip(genes[beta_idx], con)]

        self.__l_idx = sp.repeat(
          range(len_genes * len_conditions * len_sirnas_per_gene),
          len_replicates)

        self._set_link(link)
        self._set_sampler(sampler)


    @property
    def family(self):
        return self.__family

    @property
    def _beta_idx(self):
        return self.__beta_idx

    def _set_link(self, link: Link):
        if link == Link.gaussian:
            setattr(self, link, lambda x: x)
        elif link == Link.log:
            setattr(self, link, numpy.exp)
        else:
            raise ValueError("Incorrect link function specified")

    def _set_sampler(self, sampler):
        if sampler == Sampler.NUTS:
            setattr(self, sampler, pymc3.NUTS)
        elif sampler == Sampler.Metropolis:
            setattr(self, sampler, pymc3.Metropolis)
        else:
            raise ValueError("Incorect link function specified")
