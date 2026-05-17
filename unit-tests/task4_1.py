from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    res = []
    for i in range(0, len(X), 4):
        row = X[i][120:500:5]
        res.append(row)
    return res


def sum_non_neg_diag(X: List[List[int]]) -> int:
    fflag = False
    ret = 0
    for i in range(len(X[0])):
        cur = X[i][i]
        if cur >= 0:
            fflag = True
            ret += cur
    if not fflag:
        return -1
    return ret


def replace_values(X: List[List[float]]) -> List[List[float]]:
    M = deepcopy(X)
    for j in range(len(X[0])):
        medium = 0
        for i in range(len(X)):
            medium += M[i][j]
        medium /= len(X)
        for i in range(len(X)):
            if M[i][j] > 1.5 * medium or M[i][j] < medium/4:
                M[i][j] = -1
    return M
