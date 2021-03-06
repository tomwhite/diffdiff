import unittest

from reverse_autodiff import reverse_autodiff
from test_functions2 import *


class TestAutoDiff(unittest.TestCase):
    def test_reverse_autodiff_x_sq(self):
        self.assertEqual(f(1), 1)
        self.assertEqual(reverse_autodiff(f)(1), 2)

    def test_reverse_autodiff_tanh(self):
        self.assertEqual(h(1), 0.7615941559557649)
        self.assertEqual(reverse_autodiff(h)(1), 0.4199743416140261)

    def test_reverse_autodiff_x_tanh(self):
        self.assertEqual(c(1), 0.7615941559557649)
        self.assertEqual(reverse_autodiff(c)(1), 1.1815684975697909)

    def test_reverse_autodiff_sin(self):
        self.assertEqual(d(1), 0.8414709848078965)
        self.assertEqual(reverse_autodiff(d)(1), 0.5403023058681398)

    def test_two_arg_fn(self):
        self.assertEqual(g(1, 3), 7)
        self.assertEqual(reverse_autodiff(g)(1, 3), (5, 2)) # (2x+y, x+1)

if __name__ == '__main__':
    unittest.main()
