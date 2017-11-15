# Autodiff

# Use Dual objects to carry the derivative of a function around
# Simple use of Taylor expansion, dropping terms higher than epsilon:
# f(a + eps) = f(a) + f'(a) * eps 

class Dual(object):
    def __init__(self, a, b=1):
        self.a = a
        self.b = b

    def diff(self):
        return self.b

    def __mul__(self, other):
        return Dual(self.a * other.a, self.a * other.b + self.b * other.a)

    def __rmul__(self, other):
        return Dual(self.a * other.a, self.a * other.b + self.b * other.a)

    def __repr__(self):
        return "(%s, %s)" % (self.a, self.b)


def autodiff(f):
    return lambda x: f(Dual(x)).diff()


def f(x):
    return x * x


fdash = autodiff(f)

print(f(1))
print(fdash(1))


def wrap(method):
    def fn(*args, **kwargs):
        if (method.__name__ == 'tanh' and type(args[0]) == Dual):
            x = args[0].a
            return Dual(math.tanh(x), 1 / (math.cosh(x) * math.cosh(x)))  # dual calculus
        return method(*args, **kwargs)

    return fn


import math

math.tanh = wrap(math.tanh)

import math


def h(x):
    return math.tanh(x)


hdash = autodiff(h)

print(h(1))
print(hdash(1))


def c(x):
    return x * math.tanh(x)


cdash = autodiff(c)

print(c(1))
print(cdash(1))
