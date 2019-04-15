from abc import ABC

import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder

from shm.family import Family
from shm.globals import INTERVENTION, REPLICATE, GENE, CONDITION
from shm.link import Link
from shm.mixture import BinaryMixtureModel
from shm.mrf.BinaryMRF import BinaryMRF


class HM(ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 independent_interventions=False,
                 family=Family.gaussian,
                 link=Link.identity,
                 graph=None,
                 node_labels=None):

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

        if graph:
            self.__structure = BinaryMRF(graph, node_labels)
            self.__do_mrf = True
        else:
            self.__structure = BinaryMixtureModel()
            self.__do_mrf = False


    @property
    def do_random_field(self):
        return self.__do_mrf

    @property
    def family(self):
        return self.__family

    @property
    def _beta_idx(self):
        return self.__beta_idx