# encoding: utf-8
from unittest import TestCase, main as unittest_run

from data_tools.math.metrics import euclidean, manhattan


class EuclideanMetricTests(TestCase):
    """ Tests suite for euclidean distance calculation. """

    def test_different_dimensions(self):
        v1 = [1, 2]
        v2 = [0]

        self.assertRaises(ValueError, euclidean, v1, v2)

    def test_1_dimension(self):
        v1 = [10]
        v2 = [0]

        self.assertEquals(euclidean(v1, v2), 10)

    def test_3_dimension(self):
        v1 = [0, 0, 0]
        v2 = [1, 2, 3]

        self.assertEquals(euclidean(v1, v2), 3.7416573867739413)


class ManhattanMetricTests(TestCase):
    """ Tests suite for euclidean distance calculation. """

    def test_different_dimensions(self):
        v1 = [1, 2]
        v2 = [0]

        self.assertRaises(ValueError, manhattan, v1, v2)

    def test_1_dimension(self):
        v1 = [10]
        v2 = [0]

        self.assertEquals(manhattan(v1, v2), 10)

    def test_3_dimension(self):
        v1 = [0, 0, 0]
        v2 = [1, 2, 3]

        self.assertEquals(manhattan(v1, v2), 6)


if __name__ == "__main__":
    unittest_run()