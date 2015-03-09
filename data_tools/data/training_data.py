# encoding: utf-8


class TrainingData(object):
    """ Wrapper for training data.

    Holds training data structures and provides helpers for describing training data set.
    """
    def __init__(self, samples, classes):
        if not samples:
            raise ValueError("Cannot initialize TrainingData with empty samples")

        if not classes:
            raise ValueError("Cannot initialize TrainingData with empty classes")

        self._samples_count = len(samples)
        self._classes_count = len(classes)
        if self.samples_count != self._classes_count:
            raise ValueError("Cannot initialize TrainingData with samples set not matching classes set length")

        self._dimensions_count = len(samples[0])
        if not all((len(x) == self._dimensions_count for x in samples)):
            raise ValueError("Samples need to have equal dimensions")

        self.samples = samples
        self.classes = classes

    @property
    def samples_count(self):
        """ Returns number of samples in training data set. """
        return self._samples_count

    @property
    def dimensions_count(self):
        """ Returns number of dimensions of a single data set row. """
        return self._dimensions_count

    @property
    def distinct_classes(self):
        """ Returns list of distinct class names. """
        return list(set(self.classes))

    @property
    def distinct_classes_count(self):
        """ Returns number of distinct classes. """
        return len(self.distinct_classes)