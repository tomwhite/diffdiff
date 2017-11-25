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

    def __rmul__(self, other):
        return Dual(self.a * other.a, self.a * other.b + self.b * other.a)

    def __repr__(self):
        return "Dual(%r, %r)" % (self.a, self.b)


def forward_autodiff(f):
    return lambda x: f(Dual(x)).diff()

# For a better way, see https://stackoverflow.com/questions/3191799/decorate-a-whole-library-in-python

def wrap(method):
    def fn(*args, **kwargs):
        if (method.__name__ == 'tanh' and type(args[0]) == Dual):
            x = args[0].a
            return Dual(math.tanh(x), 1 / (math.cosh(x) * math.cosh(x)))  # dual calculus
        return method(*args, **kwargs)

    return fn

math.tanh = wrap(math.tanh)

