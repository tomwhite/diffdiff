import unittest

from numeric_diff import numeric_diff
from test_functions1 import *


class TestAutoDiff(unittest.TestCase):
    def test_x_sq(self):
        self.assertEqual(f(1), 1)
        self.assertAlmostEqual(numeric_diff(f)(1), 2.0, 3)  # not exactly 2

    def test_tanh(self):
        self.assertEqual(h(1), 0.7615941559557649)
        self.assertAlmostEqual(numeric_diff(h)(1), 0.4199743416140261, 3)

    def test_x_tanh(self):
        self.assertEqual(c(1), 0.7615941559557649)
        self.assertAlmostEqual(numeric_diff(c)(1), 1.1815684975697909, 3)

    def test_two_arg_fn(self):
        self.assertEqual(g(1, 3), 7)
        self.assertAlmostEqual(numeric_diff(g)(1, 3), 5.0, 3)

if __name__ == '__main__':
    unittest.main()
