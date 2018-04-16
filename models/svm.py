"""SVM.

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import numpy as np
import sklearn
import sklearn.svm
import json

import utils.datasets as ds


class SVM:
    """Anomaly detection using SVM

    Args:
        debug (bool = False): Flag to print output.
        normalize (bool = True): Normalization flag.
    """

    def __init__(self, **kwargs):
        defaults = {
            'debug': False,
            'normalize': True
        }

        vars(self).update({p: kwargs.get(p, d) for p, d in defaults.items()})

        self.model = sklearn.svm.OneClassSVM(nu=0.01, gamma=1, kernel='rbf')

    def train(self, X):
        """Train the Classifier.

        Args:
            X (np.ndarray): Features with shape
                (num_samples * time_steps, features).
        """

        # normalize X
        if self.normalize:
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)
            X = ds.rescale(X, self._min, self._max, -1, 1)

        self.model.fit(X)

    def test(self, X_test, Y):
        """Tests classifier

        Args:
            X_test (np.ndarray): Testing features.
            Y (np.array): Labels.

        Returns:
            dict: Dictionary containing the following fields:
        """
        # normalize data
        if self.normalize:
            X_test = ds.rescale(X_test, self._min, self._max, -1, 1)

        Y_pred = self.model.predict(X_test)

        data = {}
        data['confusion_matrix'] = sklearn.metrics.confusion_matrix(
            Y, Y_pred
        ).tolist()
        data['accuracy'] = sklearn.metrics.accuracy_score(Y, Y_pred) * 100

        self.print(json.dumps(data, indent=2))

        return data

    def print(self, val):
        if self.debug:
            print(val)
