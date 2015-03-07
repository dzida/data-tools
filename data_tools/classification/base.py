# encoding: utf-8


class ClassificationAlgorithmBase(object):
    """ Base class for all classification algorithms.

    Provides interface that should be implemented by each classification algorithm.
    """

    def train(self):
        raise NotImplementedError

    def classify(self):
        raise NotImplementedError