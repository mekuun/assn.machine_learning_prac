import numpy as np


def get_part_of_array(X: np.ndarray) -> np.ndarray:
    return X[::4, 120:500:5]


def sum_non_neg_diag(X: np.ndarray) -> int:
    diag_elements = np.diagonal(X)
    nn_diag_elements = diag_elements[diag_elements >= 0]
    if nn_diag_elements.size > 0:
        return np.sum(nn_diag_elements)
    else:
        return -1


def replace_values(X: np.ndarray) -> np.ndarray:
    M = X.copy()
    sr = np.mean(X, axis=0)
    M[(X < 0.25 * sr) | (X > 1.5 * sr)] = -1
    return M
