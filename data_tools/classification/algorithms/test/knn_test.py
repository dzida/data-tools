# encoding: utf-8
from unittest import TestCase, main as unittest_run

from data_tools.classification.algorithms.knn import KNearestNeighbour
from data_tools.classification.datastructures.training_data import TrainingData


class KNNClassificationAlgorithmTests(TestCase):
    """ Tests suite for base ClassificationAlgorithm tests. """

    def setUp(self):
        self.TRAINING_DATA = TrainingData([[1, 0], [0, 2], [3, 4]], ["A", "B", "C"])

    def test_k_1(self):
        K = 100
        knn = KNearestNeighbour(K)
        self.assertEquals(knn.k, K)

    def test_classify_on_not_trained(self):
        knn = KNearestNeighbour(5)
        self.assertFalse(knn.is_trained)
        self.assertRaises(ValueError, knn.classify, None)

    def test_is_trained(self):
        knn = KNearestNeighbour(1)
        knn.train(self.TRAINING_DATA)
        self.assertTrue(knn.is_trained)

    def test_k_is_greater_than_number_of_samples(self):
        K = self.TRAINING_DATA.samples_count + 1
        knn = KNearestNeighbour(K)
        self.assertRaises(ValueError, knn.train, self.TRAINING_DATA)

    def test_k_is_equal_number_of_samples(self):
        K = self.TRAINING_DATA.samples_count
        knn = KNearestNeighbour(K)
        knn.train(self.TRAINING_DATA)

        self.assertTrue(knn.is_trained)

    def test_on_training_data_k1(self):
        K = 1
        knn = KNearestNeighbour(K)
        knn.train(self.TRAINING_DATA)

        self.assertEquals(knn.classify(self.TRAINING_DATA.samples[0]).selected_class, self.TRAINING_DATA.classes[0])
        self.assertEquals(knn.classify(self.TRAINING_DATA.samples[1]).selected_class, self.TRAINING_DATA.classes[1])
        self.assertEquals(knn.classify(self.TRAINING_DATA.samples[2]).selected_class, self.TRAINING_DATA.classes[2])

    def test_results_has_all_classes(self):
        K = 1
        knn = KNearestNeighbour(K)
        knn.train(self.TRAINING_DATA)

        classification_results = knn.classify(self.TRAINING_DATA.samples[0])
        self.assertTrue(self.TRAINING_DATA.classes[1] in classification_results.results)
        self.assertTrue(self.TRAINING_DATA.classes[2] in classification_results.results)


if __name__ == "__main__":
    unittest_run()