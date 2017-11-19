def derivative(f, x, eps):
    return (f(x + eps) - f(x)) / eps


def numeric_diff(f):
    return lambda x: derivative(f, x, 0.00001)
