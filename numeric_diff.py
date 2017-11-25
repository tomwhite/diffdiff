from functools import partial


def derivative(f, eps, x):
    return (f(x + eps) - f(x)) / eps


def numeric_diff(f):
    return partial(derivative, f, 0.00001)
