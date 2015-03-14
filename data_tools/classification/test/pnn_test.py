# encoding: utf-8
from unittest import TestCase, main as unittest_run

from data_tools.classification.pnn import ProbabilisticNeuralNetwork
from data_tools.data.training_data import TrainingData


class ProbabilisticNeuralNetworkClassificationAlgorithmTests(TestCase):
    """ Tests suite for ProbabilisticNeuralNetwork tests. """

    def setUp(self):
        self.TRAINING_DATA = TrainingData([[1, 0], [0, 2], [3, 4]], ["A", "B", "C"])

    def test_sigma(self):
        SIGMA = 1
        pnn = ProbabilisticNeuralNetwork(SIGMA)
        self.assertEquals(pnn.sigma, SIGMA)

    def test_not_trained(self):
        SIGMA = 1
        pnn = ProbabilisticNeuralNetwork(SIGMA)
        self.assertFalse(pnn.is_trained)

    def test_is_trained_when_trained(self):
        SIGMA = 1
        pnn = ProbabilisticNeuralNetwork(SIGMA)
        pnn.train(self.TRAINING_DATA)
        self.assertTrue(pnn.is_trained)

    def test_on_training_data(self):
        SIGMA = 1
        pnn = ProbabilisticNeuralNetwork(SIGMA)
        pnn.train(self.TRAINING_DATA)

        self.assertEquals(pnn.classify(self.TRAINING_DATA.samples[0]), self.TRAINING_DATA.classes[0])
        self.assertEquals(pnn.classify(self.TRAINING_DATA.samples[1]), self.TRAINING_DATA.classes[1])
        self.assertEquals(pnn.classify(self.TRAINING_DATA.samples[2]), self.TRAINING_DATA.classes[2])


if __name__ == "__main__":
    unittest_run()