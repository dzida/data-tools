# encoding: utf-8
from unittest import TestCase, main as unittest_run

from data_tools.classification.knn import KNNClassificationAlgorithm


class KNNClassificationAlgorithmTests(TestCase):
    """ Tests suite for base ClassificationAlgorithm tests. """

    def test_k_1(self):
        K = 100
        knn = KNNClassificationAlgorithm(K)
        self.assertEquals(knn.k, K)

    def test_classify_on_not_trained(self):
        knn = KNNClassificationAlgorithm(5)
        self.assertFalse(knn.is_trained)
        self.assertRaises(ValueError, knn.classify, None)

    def test_is_trained(self):
        TRAINING_DATA = [1, 2]
        knn = KNNClassificationAlgorithm(1)
        knn.train(TRAINING_DATA)
        self.assertTrue(knn.is_trained)

    def test_k_is_greater_than_number_of_samples(self):
        K = 5
        training_data = [i for i in xrange(K - 1)]
        knn = KNNClassificationAlgorithm(K)
        self.assertRaises(ValueError, knn.train, training_data)

    def test_k_is_equal_number_of_samples(self):
        K = 5
        training_data = [i for i in xrange(K)]
        knn = KNNClassificationAlgorithm(K)
        knn.train(training_data)

        self.assertTrue(knn.is_trained)

    def test_on_training_data(self):
        CLASS_1 = "A"
        CLASS_2 = "B"
        CLASS_3 = "C"
        training_data = [
            [-1, 1, CLASS_1],
            [1, -1, CLASS_2],
            [0, 0, CLASS_3]
        ]

        K = 1
        knn = KNNClassificationAlgorithm(K)
        knn.train(training_data)

        self.assertEquals(knn.classify([-1, 1]), CLASS_1)
        self.assertEquals(knn.classify([1, -1]), CLASS_2)
        self.assertEquals(knn.classify([0, 0]), CLASS_3)

if __name__ == "__main__":
    unittest_run()