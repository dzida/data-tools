# encoding: utf-8
from data_tools.classification.algorithms.base import ClassificationAlgorithmBase
from data_tools.math.distance import euclidean
from data_tools.classification.datastructures.classification_results import ClassificationResults


class KNearestNeighbour(ClassificationAlgorithmBase):

    def __init__(self, k, metric=euclidean):
        self.k = k
        self._training_data = None
        self._calculate_distance = metric

    @property
    def is_trained(self):
        return self._training_data is not None

    def train(self, training_data):
        if training_data.samples_count < self.k:
            raise ValueError("k should be lesser or equal than number of samples in training data")

        # just store training data
        self._training_data = training_data

    def classify(self, data):
        if not self.is_trained:
            raise ValueError("Cannot classify on not trained classifier")

        distances = []
        for i, sample in enumerate(self._training_data.samples):
            distance = self._calculate_distance(sample, data)
            sample_class = self._training_data.classes[i]
            distances.append((distance, sample_class))
            # reverse sort after append
            distances = sorted(distances, key=lambda x: x[0])
            # trim to k
            distances = distances[:self.k]

        results = {x[1]: x[0] for x in distances}

        return ClassificationResults(results)

