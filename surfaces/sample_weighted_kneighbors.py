"""
Implement a k-nearest-neighbor classifier with support for sample weights derived
from the sklearn implementation.
"""

import numpy as np
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin
from sklearn.neighbors._base import _get_weights
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.extmath import weighted_mode

from numba import njit, prange, set_num_threads

def regularized_distance_weight(dists, p=2, mask_query=False):
    """
    Distance weight normalized to the standard deviation of the
    distances of the neighbors.
    """

    dist_med = np.median(dists, axis=-1, keepdims=True)
    # scaling by median guarantees that half of the samples will have a weight of
    # > 0.5
    weights = 1 / ((dists / dist_med) ** p + 1)
    if mask_query:
        weights[dists == 0] = 0
    return weights


def censored_uniform_weight(dists):
    """
    Distance weight that censores points where the distance is exactly zero.
    """
    weights = np.ones_like(dists)
    weights[dists == 0] = 0
    return weights


class SampleWeightedKNeighborsClassifier(
    KNeighborsMixin, ClassifierMixin, NeighborsBase
):
    """
    Classifier implementing k-nearest-neighbors vote with sample weights.

    See documentation of sklearn.neighbors.KNeighborsClassifier for description of
    arguments. In addition to the arguments supported by the sklearn verion,
    ``sample_weights`` may be provided to the ``fit`` function
    to manually weight every sample independently from the distance to the queried point.
    """

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.sample_weights = None
        self.weights = weights

    @staticmethod
    @njit(parallel=True)
    def reweight_neighbours(X, _fit_X, _y, classes, neigh_ind, query_weights=None):
        """Return re-weighting factors to remedy biases in feature distributions.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.
        _fit_X : ndarray of shape (n_samples, n_features)
            Training samples.
        _y : ndarray of shape (n_samples, )
            Training labels.
        classes : ndarray of shape (n_classes, )
            Class labels.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        query_weights : ndarray of shape (n_queries, n_neighbors), optional
            Weights of all neighbors specific to each query (such as distance weights)
        Returns
        -------
        neigh_w : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.
        """
        n_queries = X.shape[0]
        n_features = X.shape[1]

        n_neighbors = neigh_ind.shape[1]
        neigh_w = np.zeros((n_queries, n_neighbors), dtype="float64")

        for i_query in prange(n_queries):

            this_w = np.ones(n_neighbors, dtype="float64")
            neigh_inds = neigh_ind[i_query]

            for class_i in classes:

                class_mask = _y[neigh_inds] == class_i
                class_neigh_inds = neigh_inds[class_mask]

                class_neigh_sample_weights = np.ones(class_neigh_inds.shape)
                if query_weights is not None:
                    class_neigh_sample_weights *= query_weights[i_query][class_mask]

                for i_feature in range(n_features):

                    dx = (
                        _fit_X[class_neigh_inds][:, i_feature]
                        - X[:, i_feature][i_query]
                    )

                    sum_dx = np.sum(class_neigh_sample_weights * dx)
                    sum2_dx = np.sum(class_neigh_sample_weights * dx ** 2)

                    # if the dx were too small, the sum might get zero
                    # just don't re-weight this neighborhood then
                    if sum2_dx == 0.0:
                        continue

                    this_w[class_mask] *= np.exp(-sum_dx / sum2_dx * dx)

            neigh_w[i_query] = this_w

        return neigh_w

    def fit(self, X, y, sample_weights=None):
        """Fit the k-nearest neighbors classifier from the training dataset.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.
        sample_weights : array-like of shape (n_samples,)
            Sample weights.
        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """

        # TODO: Check weights are valid (_check_weights no longer exists)

        if sample_weights is not None:
            assert len(sample_weights) == len(y)
        self.sample_weights = sample_weights
        return self._fit(X, y)

    def predict(self, X, correct_bias=False, weighted_bias_correction=True):
        """Predict the class labels for the provided data.
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        correct_bias : bool
            Correct bias in feature distributions (feature-wise).
        weighted_bias_correction : bool
            If True (Default), use sample weights when calculating the bias correction.
        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        neigh_dist, neigh_ind = self.kneighbors(X)
        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = _num_samples(X)

        # These weights are a function of the neighbor distances and are different
        # for every query. The shape is (n_queries, n_neighbors).
        weights = _get_weights(neigh_dist, self.weights)
        # When the weight is configured as "uniform", the weights will be None.
        # (This is a bit silly but it comes from the base class of this class.)
        if weights is None:
            weights = np.ones(neigh_dist.shape)

        if self.sample_weights is not None:
            weights *= self.sample_weights[neigh_ind]

        if correct_bias:
            set_num_threads(self.n_jobs)
            weights *= self.reweight_neighbours(
                X,
                self._fit_X,
                self._y,
                self.classes_,
                neigh_ind,
                query_weights=weights if weighted_bias_correction else None,
            )

        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def predict_proba(self, X, correct_bias=False, weighted_bias_correction=True):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        correct_bias : bool
            Correct bias in feature distributions (feature-wise).
        weighted_bias_correction : bool
            If True (Default), use sample weights when calculating the bias correction.
        Returns
        -------
        probabilities : ndarray of shape (n_queries, n_classes), or a list of n_outputs of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(X)
        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones(neigh_dist.shape)

        if self.sample_weights is not None:
            weights *= self.sample_weights[neigh_ind]

        if correct_bias:
            set_num_threads(self.n_jobs)
            weights *= self.reweight_neighbours(
                X,
                self._fit_X,
                self._y,
                self.classes_,
                neigh_ind,
                query_weights=weights if weighted_bias_correction else None,
            )

        all_rows = np.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities

    def _more_tags(self):
        return {"multilabel": True}


def test_knn():
    from numpy.testing import assert_equal
    from sklearn.neighbors import KNeighborsClassifier

    samples = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 1.0, 0.5], [1, 2, 3]]
    sample_weights = np.array([1.0, 0.5, 0.5, 1.0])
    targets = np.array([1, 0, 0, 1])

    sw_knn = SampleWeightedKNeighborsClassifier(n_neighbors=3)
    sw_knn.fit(samples, targets, sample_weights)

    query = [[1.0, 1.0, 1.0], [0, 1, 3]]
    assert_equal(sw_knn.predict(query), [0, 0])
    # since two samples for classes 0 have the same weight as one sample from class 1,
    # the probability output should be 0.5 here
    assert_equal(sw_knn.predict_proba(query), [[0.5, 0.5], [0.5, 0.5]])

    # without sample weights, the output should be the same as that of the default KNC
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(samples, targets)
    sw_knn.fit(samples, targets)

    assert_equal(knn.predict_proba(query), sw_knn.predict_proba(query))

    # distance weights should also work the same way in the absence of sample weights

    knn = KNeighborsClassifier(n_neighbors=3, weights=regularized_distance_weight)
    knn.fit(samples, targets)

    sw_knn = SampleWeightedKNeighborsClassifier(
        n_neighbors=3, weights=regularized_distance_weight
    )
    sw_knn.fit(samples, targets)

    assert_equal(knn.predict_proba(query), sw_knn.predict_proba(query))

    print("Test passed.")


if __name__ == "__main__":
    test_knn()
