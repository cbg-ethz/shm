import pandas as pd

from shm.family import Family
from shm.link import Link
from shm.models.hm import HM
from shm.sampler import Sampler


class HGP(HM):
    def __init__(self,
                 data: pd.DataFrame,
                 family=Family.gaussian,
                 link=Link.identity,
                 graph=None,
                 node_labels=None,
                 sampler=Sampler.NUTS):
        super().__init__(data,
                         family,
                         link,
                         graph,
                         node_labels,
                         sampler)

    def _set_simple_model(self):
        raise NotImplementedError()

    def _set_mrf_model(self):
        raise NotImplementedError()

    def _set_clustering_model(self):
        raise NotImplementedError()
