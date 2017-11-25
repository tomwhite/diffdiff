import math


def f(x):
    return x * x


def fdash(x):
    return 2 * x  # do the differentiation yourself


print(f(1))
print(fdash(1))


def h(x):
    return math.tanh(x)


def hdash(x):
    return 1 / (math.cosh(x) * math.cosh(x))


print(h(1))
print(hdash(1))


def c(x):
    return x * math.tanh(x)


def cdash(x):
    return x * hdash(x) + math.tanh(x)  # using the product rule


print(c(1))
print(cdash(1))
