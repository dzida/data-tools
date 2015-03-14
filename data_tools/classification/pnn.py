# encoding: utf-8
from math import exp

from data_tools.classification.base import ClassificationAlgorithmBase
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
            sum_layer[class_] = 0.0
            for j, sample in enumerate(self._training_data.samples):
                if self._training_data.classes[j] == class_:
                    p = euclidean(self._training_data.samples[j], data) ** 2
                    sum_layer[class_] += exp(p / self._minus_variance)

            sum_layer[class_] /= self._training_data.samples_count

        # normalize
        total = sum(sum_layer.values())
        for s in sum_layer:
            sum_layer[s] /= total

        # output layer - select class
        selected = None
        largest = -1
        for s in sum_layer:
            if sum_layer[s] > largest:
                largest = sum_layer[s]
                selected = s

        return selected