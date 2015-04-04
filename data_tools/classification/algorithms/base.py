# encoding: utf-8


class ClassificationAlgorithmBase(object):
    """ Base class for all classification algorithms.

    Provides interface that should be implemented by each classification algorithm.
    """

    def train(self, training_data):
        raise NotImplementedError

    def classify(self, data):
        """ To be implemented in child classes.

        Should return ClassificationResults instance.
        """
        raise NotImplementedError