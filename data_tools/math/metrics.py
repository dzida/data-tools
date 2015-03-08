# encoding: utf-8
from numpy import linalg, array


def euclidean(v1, v2):
    if len(v1) != len(v2):
        raise ValueError(u"v1, v2 lengths must be the same")

    return linalg.norm(array(v1) - array(v2))


def manhattan(v1, v2):
    if len(v1) != len(v2):
        raise ValueError(u"v1, v2 lengths must be the same")

    return sum((abs(v[0] - v[1]) for v in zip(v1, v2)))