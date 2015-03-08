# encoding: utf-8
from functools import wraps


def assert_same_length(fn):
    """ Decorator that checks if given two vectors for distance calculation have the same length.

    Raises ValueError if length is different.
    """
    @wraps(fn)
    def wrapped(v1, v2):
        if len(v1) != len(v2):
            raise ValueError(u"v1, v2 lengths must be the same")
        return fn(v1, v2)
    return wrapped


@assert_same_length
def euclidean(v1, v2):
    """ Computes euclidean distance between two points.

    http://en.wikipedia.org/wiki/Euclidean_distance
    """
    return sum((v[0] - v[1])**2 for v in zip(v1, v2)) ** 0.5


@assert_same_length
def manhattan(v1, v2):
    """ Computes Mahattan (taxicab) distance between two points.

    http://en.wikipedia.org/wiki/Taxicab_geometry
    """
    return sum((abs(v[0] - v[1]) for v in zip(v1, v2)))


@assert_same_length
def canberra(v1, v2):
    """ Computes Canberra distance between two points.

    http://en.wikipedia.org/wiki/Canberra_distance
    """
    return sum(( (abs(v[0] - v[1]) / (abs(v[0] + abs(v[1])))) for v in zip(v1, v2)))


@assert_same_length
def chebyshev(v1, v2):
    """ Computes Chebyshev distance between two points.

    http://en.wikipedia.org/wiki/Chebyshev_distance
    """
    return max((abs(v[0] - v[1]) for v in zip(v1, v2)))