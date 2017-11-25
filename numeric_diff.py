from functools import partial


def derivative(f, eps, x):
    return (f(x + eps) - f(x)) / eps


def derivative_wrt_first_arg(f, eps, *x):
    x_plus_eps = tuple(arg + (eps if c == 0 else 0) for (c, arg) in enumerate(x))
    return (f(*x_plus_eps) - f(*x)) / eps


def numeric_diff(f):
    return partial(derivative_wrt_first_arg, f, 0.00001)
