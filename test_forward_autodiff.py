import unittest

from forward_autodiff import forward_autodiff
from test_functions1 import *


class TestAutoDiff(unittest.TestCase):
    def test_x_sq(self):
        self.assertEqual(f(1), 1)
        self.assertEqual(forward_autodiff(f)(1), 2)

    def test_tanh(self):
        self.assertEqual(h(1), 0.7615941559557649)
        self.assertEqual(forward_autodiff(h)(1), 0.4199743416140261)

    def test_x_tanh(self):
        self.assertEqual(c(1), 0.7615941559557649)
        self.assertEqual(forward_autodiff(c)(1), 1.1815684975697909)

    def test_sin(self):
        self.assertEqual(d(1), 0.8414709848078965)
        self.assertEqual(forward_autodiff(d)(1), 0.5403023058681398)


if __name__ == '__main__':
    unittest.main()
