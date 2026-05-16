import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(np.sort(x), np.sort(y))

def max_prod_mod_3(x: np.ndarray):
    p = x[1:] * x[:-1]
    p = p[p % 3 == 0]
    if len(p) == 0:
        return -1
    return p.max()


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(image, weights)

def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x_dec = np.repeat(x.T[0], x.T[1])
    y_dec = np.repeat(y.T[0], y.T[1])
    if len(x_dec) != len(y_dec):
        return -1
    return np.dot(x_dec, y_dec)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    norm = np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))
    flag = norm != 0
    cos_dist = np.ones((len(X), len(Y)))
    cos_dist[flag] = np.dot(X, Y.T)[flag] / norm[flag]
    return cos_dist