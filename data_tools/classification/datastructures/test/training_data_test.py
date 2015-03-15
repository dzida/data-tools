# encoding: utf-8
from unittest import TestCase, main as unittest_run

from data_tools.classification.datastructures.training_data import TrainingData


class TrainingDataTests(TestCase):
    """ Tests suite for training data wrapper. """
    SAMPLES = [
        [1, 2, 3, 0],
        [4, 5, 6, 0],
        [7, 8, 9, 0]
    ]

    CLASSES = [
        "A",
        "B",
        "A"
    ]

    SAMPLES_COUNT = 3
    DIMENSIONS_COUNT = 4
    CLASSES_COUNT = 2
    DISTINCT_CLASSES = ["A", "B"]

    def test_init_empty_samples(self):
        self.assertRaises(ValueError, TrainingData, [], self.CLASSES)

    def test_init_empty_classes(self):
        self.assertRaises(ValueError, TrainingData, self.SAMPLES, [])

    def test_samples_and_classes_need_to_have_same_length(self):
        self.assertRaises(ValueError, TrainingData, [1], self.CLASSES)

    def test_samples_dimensions_equality(self):
        self.assertRaises(ValueError, TrainingData, [[0], [1, 2], [1, 2, 3]], self.CLASSES)

    def test_samples_count(self):
        td = TrainingData(self.SAMPLES, self.CLASSES)
        self.assertEquals(td.samples_count, self.SAMPLES_COUNT)

    def test_dimensions_count(self):
        td = TrainingData(self.SAMPLES, self.CLASSES)
        self.assertEquals(td.dimensions_count, self.DIMENSIONS_COUNT)

    def test_distinct_classes_count(self):
        td = TrainingData(self.SAMPLES, self.CLASSES)
        self.assertEquals(td.distinct_classes_count, self.CLASSES_COUNT)

    def test_distinct_classes(self):
        td = TrainingData(self.SAMPLES, self.CLASSES)
        self.assertEquals(td.distinct_classes, self.DISTINCT_CLASSES)

    # TODO: create from numpy
if __name__ == "__main__":
    unittest_run()