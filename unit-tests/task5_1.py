from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    x_cp = sorted(x)
    y_cp = sorted(y)
    return x_cp == y_cp


def max_prod_mod_3(x: List[int]) -> int:
    mul = -1
    mulmax = -1
    for i in range(len(x) - 1):
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            mul = x[i] * x[i + 1]
            if mul > mulmax:
                mulmax = mul
    return mulmax


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    height = len(image)
    width = len(image[0])
    num_channels = len(image[0][0])
    result = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            weighted_sum_pixel = 0
            for k in range(num_channels):
                weighted_sum_pixel += image[i][j][k] * weights[k]
            result[i][j] = weighted_sum_pixel
    return result

def decode_rle(encoded_vector):
    decoded = []
    for value, count in encoded_vector:
        decoded.extend([value] * count)
    return decoded
def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    x_dec = decode_rle(x)
    y_dec = decode_rle(y)
    if len(x_dec) != len(y_dec):
        return -1

    scalar_product = sum(x_dec[i] * y_dec[i] for i in range(len(x_dec)))

    return scalar_product




def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    distances = [[0.0] * len(Y) for i in range(len(X))]
    for i in range(len(X)):
        norm_x = sum(map(lambda x: x ** 2, X[i])) ** 0.5
        for j in range(len(Y)):
            norm_y = sum(map(lambda y: y ** 2, Y[j])) ** 0.5
            if norm_x == 0 or norm_y == 0:
                distances[i][j] = 1
            else:
                d = 0.0
                for k in range(len(X[i])):
                    d += X[i][k] * Y[j][k]
                distances[i][j] = d / (norm_x * norm_y)
    return distances