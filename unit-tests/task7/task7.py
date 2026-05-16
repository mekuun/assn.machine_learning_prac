import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    res = []
    allfold = []
    lastk = 0
    for i in range(1, num_folds + 1):
        adder = np.array([])
        if i == num_folds:
            adder = np.arange(lastk, num_objects, 1)
        else:
            adder = np.arange(lastk, lastk + num_objects // num_folds)
        allfold.append(adder)
        lastk = allfold[i - 1][-1] + 1

    for i in range(num_folds):
        train = np.concatenate(allfold[:i] + allfold[i+1:])
        res.append((train, allfold[i]))

    return res


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    scores = {}
    for normalizer, normalizer_name in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:
                    sc = np.array([])
                    knn = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
                    for train, test in folds:
                        X_train, X_test = X[train], X[test]
                        y_train, y_test = y[train], y[test]
                        if normalizer is not None:
                            normalizer.fit(X_train)
                            X_train = normalizer.transform(X_train)
                            X_test = normalizer.transform(X_test)
                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(X_test)
                        sc = np.append(sc, score_function(y_test, y_pred))
                    value = np.mean(sc)
                    scores.update({(normalizer_name, n_neighbors, metric, weight): value})
    return scores
