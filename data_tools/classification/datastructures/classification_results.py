# encoding: utf-8


class ClassificationResults(object):
    """ Wrapper class for classification results data. """

    def __init__(self, results):
        self._results = results
        self._sorted_results = sorted(
            [
                (k, v) for k, v in results.iteritems()
                if v is not None
            ],
            key=lambda x: -1.0 * x[1])

    @property
    def results(self):
        return self._results

    @property
    def selected_class(self):
        return self._sorted_results[0][0]

    @property
    def selected_class_confidence(self):
        return self._sorted_results[0][1]