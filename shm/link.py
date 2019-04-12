from enum import Enum


class Link(Enum):
    identity = 1
    logit = 2
    exponential = 3

Link.identity
