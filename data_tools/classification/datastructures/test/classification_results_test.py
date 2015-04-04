# encoding: utf-8
from unittest import TestCase, main as unittest_run

from data_tools.classification.datastructures.classification_results import ClassificationResults


class ClassificationResultsTests(TestCase):
    """ Tests suite for classification results wrapper. """

    def test_selected_class(self):
        results = ClassificationResults({"Class_A": 0.8, "Class_B": 0.2})
        self.assertEquals(results.selected_class, "Class_A")

    def test_selected_class_draw(self):
        results = ClassificationResults({"Class_A": 0.5, "Class_B": 0.5})
        # selected class in case of draw does not matter (from some points of view),
        # as long as it does not change
        selected = results.selected_class
        for _ in xrange(100):
            self.assertEquals(results.selected_class, selected)

    def test_selected_class_confidence(self):
        results = ClassificationResults({"Class_A": 0.8, "Class_B": 0.2})
        self.assertEquals(results.selected_class_confidence, 0.8)


if __name__ == "__main__":
    unittest_run()