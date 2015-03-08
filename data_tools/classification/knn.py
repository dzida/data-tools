# encoding: utf-8
from data_tools.classification.base import ClassificationAlgorithmBase


class KNNClassificationAlgorithm(ClassificationAlgorithmBase):

    def __init__(self, k):
        self.k = k
        self._training_data = None

    @property
    def is_trained(self):
        return self._training_data is not None

    def train(self, training_data):
        if len(training_data) < self.k:
            raise ValueError("k should be lesser or equal than number of samples in training data")

        # just store training data
        self._training_data = training_data

    def classify(self, data):
        if not self.is_trained:
            raise ValueError("Cannot classify on not trained classifier")
