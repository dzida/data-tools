# encoding: utf-8
from unittest import TestCase, main as unittest_run

from data_tools.classification.algorithms.base import ClassificationAlgorithmBase


class BaseTests(TestCase):
    """ Tests suite for base ClassificationAlgorithm tests. """

    def test_train(self):
        self.assertRaises(NotImplementedError, ClassificationAlgorithmBase().train, None)

    def test_classify(self):
        self.assertRaises(NotImplementedError, ClassificationAlgorithmBase().classify, None)



if __name__ == "__main__":
    unittest_run()
