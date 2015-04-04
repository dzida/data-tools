# encoding: utf-8


class ClassificationResults(object):
    """ Wrapper class for classification results data. """

    def __init__(self, results):
        # store sorted results
        self._sorted_results = sorted([(k, v) for k, v in results.iteritems()], key=lambda x: -1 * x[1])

    @property
    def selected_class(self):
        return self._sorted_results[0][0]

    @property
    def selected_class_confidence(self):
        return self._sorted_results[0][1]