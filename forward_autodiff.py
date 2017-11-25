import math


# Use Dual objects to carry the derivative of a function around
# Simple use of Taylor expansion, dropping terms higher than epsilon:
# f(a + eps) = f(a) + f'(a) * eps
class Dual(object):
    def __init__(self, a, b=1.0):
        self.a = a
        self.b = b

    def diff(self):
        return self.b

    def __mul__(self, other):
        return Dual(self.a * other.a, self.a * other.b + self.b * other.a)

    def __repr__(self):
        return "Dual(%r, %r)" % (self.a, self.b)


def forward_autodiff(f):
    return lambda x: f(Dual(x)).diff()


# For a better way, see https://stackoverflow.com/questions/3191799/decorate-a-whole-library-in-python
def wrap(function):
    def fn(*args, **kwargs):
        if function in derivatives and type(args[0]) == Dual:
            x = args[0].a
            return Dual(function(x), derivatives[function](x))
        return function(*args, **kwargs)
    return fn


derivatives = {}


def register_forward_autodiff(function, derivative):
    derivatives[function] = derivative
    return wrap(function)


# Register functions and their derivatives here
math.sin = register_forward_autodiff(math.sin, math.cos)
math.tanh = register_forward_autodiff(math.tanh, lambda x: 1 / (math.cosh(x) * math.cosh(x)))
