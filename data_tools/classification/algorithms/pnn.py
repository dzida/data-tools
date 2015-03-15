# encoding: utf-8
from math import exp

from data_tools.classification.algorithms.base import ClassificationAlgorithmBase
from data_tools.math.distance import euclidean


class ProbabilisticNeuralNetwork(ClassificationAlgorithmBase):

    def __init__(self, sigma):
        self.sigma = sigma
        self._training_data = None

    @property
    def is_trained(self):
        return True if self._training_data is not None else False

    def train(self, training_data):
        self._training_data = training_data
        # pre-compute values commonly used in algorithm
        self._minus_variance = -1 * self.sigma * self.sigma

    def classify(self, data):
        sum_layer = dict()

        for class_ in self._training_data.distinct_classes:
            sum_layer[class_] = dict(score=0, count=0)

        for i, sample in enumerate(self._training_data.samples):
            sample_class = self._training_data.classes[i]
            p = euclidean(self._training_data.samples[i], data) ** 2
            sum_layer[sample_class]["score"] += exp(p / self._minus_variance)
            sum_layer[sample_class]["count"] += 1

        for s in sum_layer:
            sum_layer[s] = sum_layer[s]["score"] / sum_layer[s]["count"]

        # normalize
        total = sum(sum_layer.values())

        # output layer - select class
        selected = None
        largest = -1
        for s in sum_layer:
            # normalize
            sum_layer[s] /= total
            if sum_layer[s] > largest:
                largest = sum_layer[s]
                selected = s

        return selected