from enum import Enum

import numpy


def _id(x):
    return x


class Link(Enum):
    identity = _id
    log = numpy.exp
