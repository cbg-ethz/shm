from shm.enum import Enum
import pymc3 as pm


class Sampler(Enum):
    nuts = pm.NUTS
    metropolis = pm.Metropolis
