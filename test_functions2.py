from math import sin, tanh


def f(x):
    return x * x


def h(x):
    return tanh(x)


def c(x):
    return x * tanh(x)


def d(x):
    return sin(x)

