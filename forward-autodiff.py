# Autodiff

import math
import unittest

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
        return "(%s, %s)" % (self.a, self.b)


def forward_autodiff(f):
    return lambda x: f(Dual(x)).diff()

def wrap(method):
    def fn(*args, **kwargs):
        if (method.__name__ == 'tanh' and type(args[0]) == Dual):
            x = args[0].a
            return Dual(math.tanh(x), 1 / (math.cosh(x) * math.cosh(x)))  # dual calculus
        return method(*args, **kwargs)

    return fn

math.tanh = wrap(math.tanh)


class TestAutoDiff(unittest.TestCase):
    def test_forward_autodiff_x_sq(self):
        def f(x):
            return x * x
        self.assertEqual(f(1), 1, 0.001)
        self.assertEqual(forward_autodiff(f)(1), 2, 0.001)

    def test_forward_autodiff_tanh(self):
        def f(x):
            return math.tanh(x)
        self.assertEqual(f(1), 0.7615941559557649, 0.001)
        self.assertEqual(forward_autodiff(f)(1), 0.4199743416140261, 0.001)

    def test_forward_autodiff_x_tanh(self):
        def f(x):
            return x * math.tanh(x)
        self.assertEqual(f(1), 0.7615941559557649, 0.001)
        self.assertEqual(forward_autodiff(f)(1), 1.1815684975697909, 0.001)

if __name__ == '__main__':
    unittest.main()
