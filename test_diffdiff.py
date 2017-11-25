import math
import unittest

from forward_autodiff import forward_autodiff
from numeric_diff import numeric_diff
from reverse_autodiff import reverse_autodiff


def f(x):
    return x * x


def h(x):
    return math.tanh(x)


def c(x):
    return x * math.tanh(x)


class TestAutoDiff(unittest.TestCase):
    def test_forward_autodiff_x_sq(self):
        self.assertEqual(f(1), 1, 0.001)
        self.assertAlmostEqual(numeric_diff(f)(1), 2.0, 3)  # not exactly 2
        self.assertEqual(forward_autodiff(f)(1), 2)
        self.assertEqual(reverse_autodiff(f)(1), 2)

    def test_forward_autodiff_tanh(self):
        self.assertEqual(h(1), 0.7615941559557649)
        self.assertEqual(forward_autodiff(h)(1), 0.4199743416140261)

    def test_forward_autodiff_x_tanh(self):
        self.assertEqual(c(1), 0.7615941559557649)
        self.assertEqual(forward_autodiff(c)(1), 1.1815684975697909)


if __name__ == '__main__':
    unittest.main()
