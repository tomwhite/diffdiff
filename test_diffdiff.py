import math
from math import tanh
import unittest

from forward_autodiff import forward_autodiff
from numeric_diff import numeric_diff
from reverse_autodiff import reverse_autodiff


def f(x):
    return x * x


def h1(x):
    return math.tanh(x)


def h2(x):
    return tanh(x)


def c1(x):
    return x * math.tanh(x)


def c2(x):
    return x * tanh(x)


class TestAutoDiff(unittest.TestCase):
    def test_numeric_diff_x_sq(self):
        self.assertEqual(f(1), 1)
        self.assertAlmostEqual(numeric_diff(f)(1), 2.0, 3)  # not exactly 2

    def test_forward_autodiff_x_sq(self):
        self.assertEqual(f(1), 1)
        self.assertEqual(forward_autodiff(f)(1), 2)

    def test_reverse_autodiff_x_sq(self):
        self.assertEqual(f(1), 1)
        self.assertEqual(reverse_autodiff(f)(1), 2)

    def test_numeric_diff_tanh(self):
        self.assertEqual(h1(1), 0.7615941559557649)
        self.assertAlmostEqual(numeric_diff(h1)(1), 0.4199743416140261, 3)

    def test_forward_autodiff_tanh(self):
        self.assertEqual(h1(1), 0.7615941559557649)
        self.assertEqual(forward_autodiff(h1)(1), 0.4199743416140261)

    def test_reverse_autodiff_tanh(self):
        self.assertEqual(h2(1), 0.7615941559557649)
        self.assertEqual(forward_autodiff(h1)(1), 0.4199743416140261)
        self.assertEqual(reverse_autodiff(h2)(1), 0.4199743416140261)

    def test_numeric_diff_x_tanh(self):
        self.assertEqual(c1(1), 0.7615941559557649)
        self.assertAlmostEqual(numeric_diff(c1)(1), 1.1815684975697909, 3)

    def test_forward_autodiff_x_tanh(self):
        self.assertEqual(c1(1), 0.7615941559557649)
        self.assertEqual(forward_autodiff(c1)(1), 1.1815684975697909)

    def test_reverse_autodiff_x_tanh(self):
        self.assertEqual(c2(1), 0.7615941559557649)
        self.assertEqual(reverse_autodiff(c2)(1), 1.1815684975697909)


if __name__ == '__main__':
    unittest.main()
