import ast
import inspect
import re
import sympy


def find_arg_names(f):
    '''Returns a list of argument names for a function.'''
    src = inspect.getsource(f)
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                names.append(arg.arg)
    return names


def find_function_expression(f):
    '''Returns the expression defining f as a string.
    Only works for single-line functions with basic arithmetic that don't call other functions.
    '''
    ret_line = inspect.getsourcelines(f)[0][-1]  # get final line
    s = re.sub('^return', '', ret_line.strip())  # remove 'return' from start of line
    return s.strip()


def symbolic_diff(f):
    '''Returns a function that is the derivative of f.
    This uses SymPy, so it can only differentiate functions that SymPy knows about.
    E.g. tanh(x) will work, but math.tanh(x) will not.'''
    args = find_arg_names(f)
    expr = find_function_expression(f)
    s = sympy.sympify(expr)  # turn Python expression into SymPy expression
    d = sympy.diff(s, args[0])  # differentiate w.r.t first arg
    symbols = sympy.symbols(' '.join(args))  # turn all arguments into SymPy symbols
    return sympy.lambdify(symbols, d, "math")  # turn SymPy expression into Python lambda function
