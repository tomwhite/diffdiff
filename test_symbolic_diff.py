import unittest

from symbolic_diff import symbolic_diff, find_arg_names, find_function_expression
from test_functions2 import *


class TestAutoDiff(unittest.TestCase):
    def test_x_sq(self):
        self.assertEqual(f(1), 1)
        self.assertEqual(symbolic_diff(f)(1), 2)

    def test_tanh(self):
        self.assertEqual(h(1), 0.7615941559557649)
        self.assertAlmostEqual(symbolic_diff(h)(1), 0.4199743416140261, 15)

    def test_x_tanh(self):
        self.assertEqual(c(1), 0.7615941559557649)
        self.assertAlmostEqual(symbolic_diff(c)(1), 1.1815684975697909, 15)

    def test_sin(self):
        self.assertEqual(d(1), 0.8414709848078965)
        self.assertEqual(symbolic_diff(d)(1), 0.5403023058681398)

    def test_two_arg_fn(self):
        self.assertEqual(g(1, 3), 7)
        self.assertEqual(symbolic_diff(g)(1, 3), 5)

    def test_find_arg_names(self):
        self.assertEqual(find_arg_names(f), ['x'])

    def test_find_function_expression(self):
        self.assertEqual(find_function_expression(f), 'x * x')


if __name__ == '__main__':
    unittest.main()
