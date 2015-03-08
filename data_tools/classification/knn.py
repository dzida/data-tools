# encoding: utf-8
from data_tools.classification.base import ClassificationAlgorithmBase
from data_tools.math.distance import euclidean


class KNNClassificationAlgorithm(ClassificationAlgorithmBase):

    def __init__(self, k, metric=euclidean):
        self.k = k
        self._training_data = None
        self._calculate_distance = metric

    @property
    def is_trained(self):
        return self._training_data is not None

    def train(self, training_data):
        if len(training_data) < self.k:
            raise ValueError("k should be lesser or equal than number of samples in training data")

        # just store training data
        self._training_data = training_data

    def _select_winner(self, distances):
        results = dict()
        the_best = -1
        winner = None

        for distance, class_ in distances:
            if class_ not in results:
                results[class_] = 0
            else:
                results[class_] += 1

            if results[class_] > the_best:
                winner = class_
        return winner

    def classify(self, data):
        if not self.is_trained:
            raise ValueError("Cannot classify on not trained classifier")

        distances = []
        for training_sample in self._training_data:
            distance = self._calculate_distance(training_sample[:-1], data)
            sample_class = training_sample[-1]
            distances.append((distance, sample_class))
            # reverse sort after append
            distances = sorted(distances, key=lambda x: x[0])
            # trim to k
            distances = distances[:self.k]

        return self._select_winner(distances)

