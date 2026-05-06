import math


def hal(index, base):
    result = 0
    factor = 1.0 / base
    value = index

    while value > 0:
        result += factor * (value % base)
        value = math.floor(value / base)
        factor = factor / base

    return result
