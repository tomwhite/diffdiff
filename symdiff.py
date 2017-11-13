import ast
import inspect		
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
	
import re
def find_function_expression(f):
	'''Returns the expression defining f as a string.
	Only works for single-line functions with basic arithmetic that don't call other functions.
	'''
	ret_line = inspect.getsourcelines(f)[0][-1] # get final line
	s = re.sub('^return', '', ret_line.strip()) # remove 'return' from start of line
	return s.strip()
	
import sympy
def symbolic_diff(f):
	vars = find_arg_names(f)
	expr = find_function_expression(f)
	s = sympy.sympify(expr) # turn Python expression into SymPy expression
	d = sympy.diff(s, vars[0]) # differentiate w.r.t first arg
	syms = sympy.symbols(' '.join(vars)) # turn all arguments into SymPy symbols
	return sympy.lambdify(syms, d, "math") # turn SymPy expression into Python lambda function
	
## Examples

def f(x):
	return x * x

def g(x, y):
	return x * x + x * y + y

from math import tanh
def h(x):
	return tanh(x) # this only works because SymPy has a tanh function too; math.tanh(x) fails


print(find_arg_names(f))
print(find_arg_names(g))
print(find_function_expression(f))
print(find_function_expression(g))

fdash = symbolic_diff(f)
print(fdash(1))

gdash = symbolic_diff(g)
print(gdash(1, 3)) # 2*x + y

hdash = symbolic_diff(h)
print(hdash(1))